import cv2
import threading
import time
from collections import deque
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

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
            capture_thread.start()

    def _capture_frames(self, stream_id):
        while self.running:
            with self.lock:
                if stream_id not in self.video_streams:
                    break
                
                cap = self.video_streams[stream_id]['stream']
                priority = self.video_streams[stream_id]['priority']
                
                success, frame = cap.read()
                if success:
                    self.buffer.append((stream_id, frame))  # Store (stream_id, frame)
            
            time.sleep(1 / priority)  # Rate-limit based on priority

    def get_frame(self):
        with self.lock:
            if not self.buffer:
                raise HTTPException(status_code=500, detail="No frames available")
            
            stream_id, frame = self.buffer[-1]  # Get the latest frame from buffer
            _, encoded_frame = cv2.imencode('.jpg', frame)
            return {"stream_id": stream_id, "frame": encoded_frame.tobytes()}

    def update_priority(self, stream_id: int, new_priority: int):
        with self.lock:
            if stream_id in self.video_streams:
                self.video_streams[stream_id]['priority'] = new_priority
                return {"success": True}
        raise HTTPException(status_code=404, detail="Stream ID not found")

handler = VideoStreamHandler()

@app.post("/add_stream/")
def add_stream(stream_id: int, source: str, priority: int):
    try:
        handler.add_video_stream(stream_id, source, priority)
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_frame/")
def get_frame():
    return handler.get_frame()

@app.put("/update_priority/")
def update_priority(stream_id: int, new_priority: int):
    return handler.update_priority(stream_id, new_priority)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
