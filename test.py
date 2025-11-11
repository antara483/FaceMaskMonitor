import pyttsx3
import threading
import time

# ----------------------------
# Voice Engine Init (same as main script)
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

def test_tts():
    print("Testing TTS system...")
    speak_text_nonblocking("Test message for mask detection system")
    print("Message spoken - waiting 3 seconds to complete...")
    time.sleep(3)  # Wait for speech to complete
    print("TTS test completed!")

# Test different alert messages
def test_all_alerts():
    test_messages = [
        "Please wear your mask properly",
        "You still haven't worn your mask properly", 
        "Please adjust your mask to cover nose and mouth",
        "Mask detection system is working correctly"
    ]
    
    for i, message in enumerate(test_messages):
        print(f"Test {i+1}: {message}")
        speak_text_nonblocking(message)
        time.sleep(2)  # Wait between messages

if __name__ == "__main__":
    test_tts()
    print("\n" + "="*50)
    test_all_alerts()