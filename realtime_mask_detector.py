import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import deque, Counter
import time
import pyttsx3
import threading

GLOBAL_ALERT_COOLDOWN = 5  # seconds between repeated alerts globally

# ----------------------------
# Config
# ----------------------------
# Alert settings
ALERT_COOLDOWN = 3.0   # seconds between alerts for same person
REMINDER_INTERVAL = 6.0  # seconds before repeating reminder if still no mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/mask_detector.pth"
IMG_SIZE = 128
MIN_FACE_SIZE = 40



# Decision tuning
MEAN_PROB_THRESH = 0.65
MEAN_MARGIN = 0.20
IOU_MATCH_THRESHOLD = 0.4
HISTORY_LEN = 10
TRACK_DECAY_FRAMES = 12

# Preprocessing / CLAHE
USE_CLAHE = True
CLAHE_CLIP = 2.0
PAD_RATIO = 0.30



# ----------------------------
# Voice Engine Init
# ----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)   # Speed
engine.setProperty('volume', 1.0) # Volume
_tts_lock = threading.Lock()

def speak_text_nonblocking(text: str):
    """Speak text in a background thread while ensuring only one TTS call runs at a time."""
    def _run():
        with _tts_lock:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                # If TTS fails, ignore but print for debugging
                print("TTS error:", e)
    t = threading.Thread(target=_run, daemon=True)
    t.start()

# ----------------------------
# Load model
# ----------------------------
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------------------
# Transforms (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# Face detector (SSD)
# ----------------------------
face_model = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)
labels = ["With Mask", "Without Mask"]
colors = [(0, 255, 0), (0, 0, 255)]

# ----------------------------
# Tracking structures
# ----------------------------
tracks = {}
next_track_id = 0
frame_idx = 0
mask_count = 0
no_mask_count = 0

# ----------------------------
# Helpers
# ----------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1); interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1))
    boxBArea = max(1, (boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1))
    denom = float(boxAArea + boxBArea - interArea)
    return interArea/denom if denom > 0 else 0

def pad_bbox(x1, y1, x2, y2, w, h, pad_ratio=PAD_RATIO):
    bw = x2 - x1; bh = y2 - y1
    pad_w = int(bw * pad_ratio); pad_h = int(bh * pad_ratio)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - int(pad_h * 0.6))
    nx2 = min(w-1, x2 + pad_w)
    ny2 = min(h-1, y2 + pad_h)
    return int(nx1), int(ny1), int(nx2), int(ny2)

def apply_clahe_to_face(face_bgr):
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    brightness_std = np.std(y)
    if brightness_std < 45:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
        y = clahe.apply(y)
        ycrcb = cv2.merge((y, cr, cb))
        face_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return face_bgr

def mean_probs_from_history(prob_hist):
    if len(prob_hist) == 0:
        return None
    return np.mean(np.array(prob_hist), axis=0)

# ----------------------------
# Main loop
# ----------------------------
global_last_alert_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    detections = face_model.forward()

    det_boxes, det_confs = [], []

    # Face detection
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > 0.5:
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
            if x2 - x1 < MIN_FACE_SIZE or y2 - y1 < MIN_FACE_SIZE:
                continue
            det_boxes.append((x1, y1, x2, y2))
            det_confs.append(conf)

    # Predict faces
    current_faces = []
    for (x1, y1, x2, y2), det_conf in zip(det_boxes, det_confs):
        px1, py1, px2, py2 = pad_bbox(x1, y1, x2, y2, w, h)
        face_crop = frame[py1:py2, px1:px2]
        if face_crop.size == 0:
            continue

        if USE_CLAHE:
            face_crop = apply_clahe_to_face(face_crop)

        fh, fw = face_crop.shape[:2]
        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            current_faces.append({'bbox': (px1, py1, px2, py2),
                                  'probs': np.array([0.5, 0.5]), 'quality': 'low'})
            continue

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(input_tensor), dim=1).cpu().numpy()[0]
        current_faces.append({'bbox': (px1, py1, px2, py2), 'probs': probs, 'quality': 'good'})

    # Associate detections with tracks
    for det in current_faces:
        best_tid, best_iou = None, 0
        for tid, tr in tracks.items():
            i = iou(det['bbox'], tr['bbox'])
            if i > best_iou:
                best_iou, best_tid = i, tid
        if best_iou > IOU_MATCH_THRESHOLD and best_tid is not None:
            tr = tracks[best_tid]
            tr['bbox'] = det['bbox']
            tr['last_seen'] = frame_idx
            tr['probs_history'].append(det['probs'])
        else:
            tracks[next_track_id] = {
                'bbox': det['bbox'],
                'last_seen': frame_idx,
                'probs_history': deque([det['probs']], maxlen=HISTORY_LEN),
                'counted': False,
                'first_seen': frame_idx,
                'last_alert': 0.0,  # timestamp of last TTS alert for this person
                'first_no_mask_time': None  # add this line
            }
            next_track_id += 1

    # Cleanup and finalize counts
    remove_ids = []
    for tid, tr in list(tracks.items()):
        if tr['last_seen'] < frame_idx - TRACK_DECAY_FRAMES:
            mean_probs = mean_probs_from_history(tr['probs_history'])
            if mean_probs is not None:
                top_idx = int(np.argmax(mean_probs))
                top_prob = float(mean_probs[top_idx])
                runner_up = float(mean_probs[1 - top_idx])
                if top_prob >= MEAN_PROB_THRESH and (top_prob - runner_up) >= MEAN_MARGIN:
                    if not tr['counted']:
                        if top_idx == 0:
                            mask_count += 1
                        else:
                            no_mask_count += 1
                        tr['counted'] = True
            remove_ids.append(tid)
    for rid in remove_ids:
        del tracks[rid]
 
    # Visualization + per-track alerting
    mask_conf_list, nomask_conf_list = [], []
    current_time = time.time()
    for tid, tr in tracks.items():
        x1, y1, x2, y2 = tr['bbox']
        mean_probs = mean_probs_from_history(tr['probs_history'])
        label_text = "Analyzing"
        color = (200, 200, 0)
        if mean_probs is not None:
            mask_conf_list.append(mean_probs[0])
            nomask_conf_list.append(mean_probs[1])
            top_idx = int(np.argmax(mean_probs))
            top_prob = float(mean_probs[top_idx])
            margin = top_prob - float(mean_probs[1 - top_idx])
            if top_prob >= MEAN_PROB_THRESH and margin >= MEAN_MARGIN:
                label_text = labels[top_idx]
                color = colors[top_idx]
         
                    
                if labels[top_idx] == "Without Mask":
                    last_alert = tr.get('last_alert', 0.0)
                    first_no_mask_time = tr.get('first_no_mask_time', None)

    # If this is the first time detected without mask
                    if first_no_mask_time is None:
                        tr['first_no_mask_time'] = current_time
                        speak_text_nonblocking("Please wear your mask")
                        tr['last_alert'] = current_time

    # If already seen without mask, check if 6 sec passed since last alert
                    # elif current_time - tr['first_no_mask_time'] >= 6.0 and current_time - last_alert >= 6.0:
                    elif current_time - tr['first_no_mask_time'] >= REMINDER_INTERVAL and current_time - last_alert >= REMINDER_INTERVAL:
                        speak_text_nonblocking("You still haven't worn your mask ")
                        tr['last_alert'] = current_time

                else:
    # If person now wears a mask, reset timers
                    tr['first_no_mask_time'] = None
                    tr['last_alert'] = 0.0
                
                
  
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{tid} {label_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



    # Real-time Info Display
    mean_mask_conf = np.mean(mask_conf_list) if mask_conf_list else 0
    mean_nomask_conf = np.mean(nomask_conf_list) if nomask_conf_list else 0

    cv2.rectangle(frame, (10, 10), (380, 115), (0, 0, 0), -1)
    cv2.putText(frame, f"Faces in Frame: {len(tracks)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Mean WithMask Conf: {mean_mask_conf*100:.1f}%", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Mean NoMask Conf: {mean_nomask_conf*100:.1f}%", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Real-Time Mask Detector (v4 + Optimized Voice Alert)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


