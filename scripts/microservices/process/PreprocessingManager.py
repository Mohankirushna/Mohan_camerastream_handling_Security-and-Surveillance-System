from collections import deque
import threading
import requests
import numpy as np
import cv2
import time
import base64

from logger import logger
from FrameProcessor import FrameProcessor

class PreprocessingManager:
    def __init__(self, frame_processor: FrameProcessor, pull_url: str, buffer_size=10):
        self.frame_processor = frame_processor
        self.processed_buffer = deque(maxlen=buffer_size)
        self.pull_url = pull_url
        self.lock = threading.Lock()
        self.running = True
        
        self.pull_thread = threading.Thread(target=self.pull_and_process_frames, daemon=True)
        self.pull_thread.start()

    def pull_and_process_frames(self):
        session = requests.Session()
        while self.running:
            try:
                response = session.get(self.pull_url)
                if response.status_code == 200:
                    frame_data = response.json()
                    stream_id = frame_data["stream_id"]

                    decoded = base64.b64decode(frame_data["frame"])
                    frame_array = np.frombuffer(decoded, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    processed_frame = self.frame_processor.process_frame(frame)
                    # Encode frames for transmission
                    encoded_frames = {k:(cv2.imencode(".jpg", v)[-1]) for k,v in processed_frame.items()}
                    encoded_frames_b64 = {k:base64.b64encode(v.tobytes()).decode('utf-8') for k,v in encoded_frames.items()}

                    encoded_frames_b64['stream_id'] = stream_id
                    
                    with self.lock:
                        print(encoded_frames_b64['stream_id'])
                        self.processed_buffer.append(encoded_frames_b64)

            except requests.RequestException as e:
                logger.error(f"Error pulling frames: {e}")
            time.sleep(0.01)  # Prevent excessive requests

    def get_latest_frame(self):
        with self.lock:
            if self.processed_buffer:
                return self.processed_buffer.popleft()  # Get the latest processed frame
            return None