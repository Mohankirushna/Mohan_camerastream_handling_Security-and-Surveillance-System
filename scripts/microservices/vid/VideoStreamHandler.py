import threading
import cv2
from collections import deque
import time
from fastapi import HTTPException
import base64
import numpy as np
from skimage.metrics import structural_similarity as ssim
import uuid
from datetime import datetime

from logger import logger, LOG_IMAGE_DIR

def save_fault_image(stream_id, frame, fault_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{LOG_IMAGE_DIR}/{stream_id}_{fault_type}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    cv2.imwrite(filename, frame)
    logger.info(f"Saved fault image for stream {stream_id}: {filename}")

class VideoStreamHandler:
    def __init__(self, buffer_size=10):
        self.video_streams = {}  # stream_id -> {stream, priority}
        self.buffer = deque(maxlen=buffer_size)  # Shared buffer for all streams
        self.lock = threading.Lock()
        self.running = True
        self.capture_threads = {}

    def add_video_stream(self, stream_id: int, source: str, priority: int):
        with self.lock:
            if stream_id in self.video_streams:
                raise ValueError("Stream ID already exists")
            
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError("Cannot open video source")
            
            self.video_streams[stream_id] = {'stream': cap, 'priority': priority}
            capture_thread = threading.Thread(target=self._capture_frames, args=(stream_id,), daemon=True)
            self.capture_threads[stream_id] = capture_thread

    def start_all(self):
        for stream_id,capture_thread in self.capture_threads.items():
            capture_thread.start()

    def _capture_frames(self, stream_id):
        prev_frame = None

        while self.running:
            if stream_id not in self.video_streams:
                break

            cap = self.video_streams[stream_id]['stream']
            priority = self.video_streams[stream_id]['priority']

            success, frame = cap.read()
            if not success:
                logger.warning(f"Stream {stream_id} interrupted (read failure)")
                time.sleep(1)
                continue

            # Fault Detection
            fault = False
            if self.is_black_frame(frame):
                logger.warning(f"[ALERT] Stream {stream_id} - Black frame detected")
                save_fault_image(stream_id, frame, "Black")
                fault = True

            if self.is_blurry(frame):
                logger.warning(f"[ALERT] Stream {stream_id} - Blurry frame detected")
                save_fault_image(stream_id, frame, "Blurry")
                fault = True

            if prev_frame is not None and self.is_frozen_frame(prev_frame, frame):
                logger.warning(f"[ALERT] Stream {stream_id} - Frozen frame detected")
                save_fault_image(stream_id, frame, "Frozen")
                fault = True

            prev_frame = frame.copy()
            if not fault:
                with self.lock:
                    self.buffer.append((stream_id, frame))

            time.sleep(1 / priority)

    def get_frame(self):
            if not self.buffer:
                raise HTTPException(status_code=500, detail="No frames available")
            with self.lock:
                stream_id, frame = self.buffer.popleft()  # Get the latest frame from buffer
            _, encoded_frame = cv2.imencode('.jpg', frame)
            encoded_b64 = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')
            return {"stream_id": stream_id, "frame": encoded_b64}

    def update_priority(self, stream_id: int, new_priority: int):
        with self.lock:
            if stream_id in self.video_streams:
                self.video_streams[stream_id]['priority'] = new_priority
                return {"success": True}
        raise HTTPException(status_code=404, detail="Stream ID not found")
    
    def is_blurry(self, frame, threshold=100.0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    
    def is_black_frame(self, frame, threshold=10):
        return np.mean(frame) < threshold

    def is_frozen_frame(self, prev_frame, curr_frame, threshold=0.98):
        grayA = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(grayA, grayB, full=True)
        return score > threshold

