import cv2
import threading
import time
import requests
from collections import deque
from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np

VIDEO_STREAM_SERVICE_URL = "http://videostream-service:8000/get_frame/"

app = FastAPI()

class FrameProcessor:
    def process_frame(self, frame):
        # Dummy processing (split into scene and object frames)
        scene_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Simulated scene processing
        object_frame = cv2.Canny(frame, 100, 200)  # Simulated object processing

        return {
            "scene_frame": scene_frame,
            "object_frame": object_frame
        }

class PreprocessingManager:
    def __init__(self, frame_processor: FrameProcessor, buffer_size=10):
        self.frame_processor = frame_processor
        self.processed_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True
        
        self.pull_thread = threading.Thread(target=self.pull_and_process_frames, daemon=True)
        self.pull_thread.start()

    def pull_and_process_frames(self):
        while self.running:
            try:
                response = requests.get(VIDEO_STREAM_SERVICE_URL)
                if response.status_code == 200:
                    frame_data = response.json()
                    stream_id = frame_data["stream_id"]
                    frame_array = np.frombuffer(bytes(frame_data["frame"]), dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    processed_frame = self.frame_processor.process_frame(frame)

                    # Encode frames for transmission
                    _, scene_encoded = cv2.imencode(".jpg", processed_frame["scene_frame"])
                    _, object_encoded = cv2.imencode(".jpg", processed_frame["object_frame"])

                    with self.lock:
                        self.processed_buffer.append({
                            "stream_id": stream_id,
                            "scene_frame": scene_encoded.tobytes(),
                            "object_frame": object_encoded.tobytes()
                        })

            except requests.RequestException as e:
                print(f"Error pulling frames: {e}")
            time.sleep(0.01)  # Prevent excessive requests

    def get_latest_frame(self):
        with self.lock:
            if self.processed_buffer:
                return self.processed_buffer[-1]  # Get the latest processed frame
            return None

@app.get("/get_processed_frame/")
def get_processed_frame():
    frame_data = preprocessing_manager.get_latest_frame()
    if frame_data:
        return {
            "stream_id": frame_data["stream_id"],
            "scene_frame": frame_data["scene_frame"],
            "object_frame": frame_data["object_frame"]
        }
    raise HTTPException(status_code=404, detail="No processed frame available")

frame_processor = FrameProcessor()
preprocessing_manager = PreprocessingManager(frame_processor)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
