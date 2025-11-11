# FaceMaskMonitor

FaceMaskMonitor is a real-time face mask detection system that uses deep learning and computer vision to detect 
whether a person is wearing a mask or not. The system also provides voice alerts for people not wearing masks and tracks mask usage statistics over time.

## Technology Stack

- Python 3.x

- PyTorch for deep learning inference

- OpenCV for real-time video capture, face detection, and visualization

- Pillow for image preprocessing

- NumPy for numerical operations

- pyttsx3 for text-to-speech voice alerts
## Features

- Real-time detection using webcam or live video feed

-  Face detection via SSD (Single Shot MultiBox Detector) using OpenCV DNN module

- Mask classification using a trained MobileNetV2 model

- Per-person tracking to avoid repeated alerts for the same person

- Voice alerts using pyttsx3 when a person is not wearing a mask

- Statistics tracking: counts of faces present , average confidence levels

## How It Works

###  1. Face Detection:

The system uses a pre-trained SSD model (deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel) to detect faces in each frame.

### 2. Face Preprocessing:

Each detected face is padded, resized, and optionally enhanced using CLAHE for better performance under varying lighting conditions.

### 3. Mask Classification:

Faces are passed through a MobileNetV2-based classifier (mask_detector.pth) to predict With Mask or Without Mask.

Predictions are smoothed using a probability history to improve stability.

### 4. Tracking & Alerts:

Detected faces are tracked over frames.

First-time Without Mask detection triggers a voice alert.

### 5. Visualization:

Bounding boxes with mask labels are drawn on the video feed.

Real-time statistics like mean mask/no-mask confidence and number of faces detected are displayed.

## Applications

 1.Public Spaces Monitoring: Airports, train stations, offices, hospitals

 2.Workplace Safety Compliance: Ensures mask usage among employees

 3.Educational Institutions: Monitors compliance in classrooms and labs

 4.Custom Deployments: Can be integrated with IoT devices, access control systems, or dashboards

## Future Improvements

 1.Multi-camera support for large spaces

 2.Cloud-based analytics dashboard with alerts and historical statistics

 3.Lightweight mobile deployment for edge devices






