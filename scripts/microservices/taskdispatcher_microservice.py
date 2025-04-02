import cv2
import threading
import time
import requests
from collections import deque
from fastapi import FastAPI, HTTPException
import uvicorn
from concurrent.futures import ThreadPoolExecutor

PREPROCESSING_SERVICE_URL = "http://preprocessing-service:8001/get_processed_frame/"

class BaseModel:
    def __init__(self):
        self.model = None
        self.out = {}

    def detect(self, frame):
        # Simulated detection output
        output: dict[str, list[dict[str, any]]] = {'detections': [{'object': 'dummy', 'confidence': 0.9}]}
        self.out = output
        return output

class TaskDispatcher:
    def __init__(self, dispatch_models: dict[str, BaseModel], buffer_size=10, max_threads=4):
        self.dispatching_models = dispatch_models
        self.result_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True

        # Thread pool for parallel model execution
        self.executor = ThreadPoolExecutor(max_threads)

        self.dispatch_thread = threading.Thread(target=self.fetch_and_dispatch, daemon=True)
        self.dispatch_thread.start()

    def fetch_and_dispatch(self):
        while self.running:
            try:
                response = requests.get(PREPROCESSING_SERVICE_URL)
                if response.status_code == 200:
                    frame_data = response.json()
                    stream_id = frame_data["stream_id"]

                    # Decode frames dynamically
                    frames = {}
                    for key, encoded_frame in frame_data.items():
                        if key != "stream_id":  # Skip stream ID
                            frames[key] = cv2.imdecode(bytearray(encoded_frame), cv2.IMREAD_COLOR)

                    results = self.dispatch(frames)

                    compiled_result = {
                        "stream_id": stream_id,
                        "results": results
                    }

                    with self.lock:
                        self.result_buffer.append(compiled_result)

            except requests.RequestException as e:
                print(f"Error fetching processed frames: {e}")

            time.sleep(0.01)  # Prevent excessive requests

    def dispatch(self, frames: dict[str, any]):
        results = {}

        futures = []
        for frame_type, frame in frames.items():
            if frame_type in self.dispatching_models:
                model = self.dispatching_models[frame_type]
                futures.append(self.executor.submit(self.run_model, model, frame_type, frame))

        # Collect results from all models
        for future in futures:
            frame_type, out = future.result()
            results[frame_type] = out

        return results

    def run_model(self, model, frame_type, frame):
        return frame_type, model.detect(frame)

    def get_latest_result(self):
        with self.lock:
            if self.result_buffer:
                return self.result_buffer[-1]  # Get the latest compiled result
            return None

# Initialize models dynamically
models = {
    "scene_frame": BaseModel(),
    "object_frame": BaseModel(),
    "pose_frame": BaseModel(),  # Example: Human pose detection
    "emotion_frame": BaseModel(),  # Example: Emotion recognition
}

task_dispatcher = TaskDispatcher(models)

app = FastAPI()

@app.get("/get_results/")
def get_results():
    result = task_dispatcher.get_latest_result()
    if result:
        return result
    raise HTTPException(status_code=404, detail="No results available")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
