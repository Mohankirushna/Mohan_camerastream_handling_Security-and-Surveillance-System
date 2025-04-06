import threading
import cv2
from collections import deque
import time
from fastapi import HTTPException
import base64
import numpy as np

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
        while self.running:
            if stream_id not in self.video_streams:
                break
            
            cap = self.video_streams[stream_id]['stream']
            priority = self.video_streams[stream_id]['priority']
            
            success, frame = cap.read()
            if success:
                with self.lock:
                    curr = list(self.buffer)
                    print("Queue", [i[0] for i in curr])
                    self.buffer.append((stream_id, frame))  # Store (stream_id, frame)
            
            time.sleep(1 / priority)  # Rate-limit based on priority

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