from concurrent.futures import ThreadPoolExecutor
import cv2
import threading
import time
import requests
from collections import deque
import base64
import numpy as np

from BufferItem import BufferItem
from BaseModel import BaseModel

class TaskDispatcher:
    def __init__(self, dispatch_models: dict[str, BaseModel], pull_url, buffer_size=10):
        self.dispatching_models = dispatch_models
        self.result_buffer = deque(maxlen=buffer_size)
        self.pull_url = pull_url
        self.lock = threading.Lock()
        self.running = True

        # Thread pool for parallel model execution
        max_threads = len(dispatch_models)
        self.executor = ThreadPoolExecutor(max_threads)

        self.dispatch_thread = threading.Thread(target=self.fetch_and_dispatch, daemon=True)
        self.dispatch_thread.start()

    def fetch_and_dispatch(self):
        session = requests.Session()
        while self.running:
            try:
                response = session.get(self.pull_url)
                if response.status_code == 200:
                    frame_data = response.json()
                    stream_id = frame_data["stream_id"]
                    scene_frame = frame_data["scene_frame"]
                    frames = {}
                    for key, encoded_frame in frame_data.items():
                        if key not in ("stream_id", "scene_frame"):
                            decoded = base64.b64decode(encoded_frame)
                            frame_array = np.frombuffer(decoded, dtype=np.uint8)
                            frames[key] = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    t0 = time.time()
                    results = self.dispatch(frames)
                    print("dispatching time: ",time.time()-t0)
                    new_buffer_item = BufferItem(stream_id,results,scene_frame)

                    with self.lock:
                        self.result_buffer.append(new_buffer_item)
                        print(new_buffer_item.__dict__()['results'], len(self.result_buffer))

            except requests.RequestException as e:
                print(f"Error fetching processed frames: {e}")

            time.sleep(0.01)

    def dispatch(self, frames: dict[str, any]):
        results = {}

        futures = []
        for frame_type, frame in frames.items():
            if frame_type in self.dispatching_models:
                model = self.dispatching_models[frame_type]
                futures.append(self.executor.submit(self.run_model, model, frame_type, frame))

        for future in futures:
            frame_type, out = future.result()
            results[frame_type] = out

        return results

    def run_model(self, model: BaseModel, frame_type: str, frame):
        return frame_type, model.detect(frame)['detections']

    def get_latest_result(self)->BufferItem|None:
        with self.lock:
            if self.result_buffer:
                return self.result_buffer.popleft()
            return None
