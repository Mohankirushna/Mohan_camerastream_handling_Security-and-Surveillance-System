import time
import requests
import threading

class DecisionEngine:
    def __init__(self, pull_url, trigger_url, polling_interval=0.5):
        self.polling_interval = polling_interval
        self.pull_url = pull_url
        self.trigger_url = trigger_url
        self.running = True

    def start(self):
        threading.Thread(target=self.poll_task_dispatcher, daemon=True).start()

    def poll_task_dispatcher(self):
        session = requests.Session()
        while self.running:
            try:
                response = session.get(self.pull_url)
                if response.status_code == 200:
                    result = response.json()
                    self.process_result(result)
                
            except requests.RequestException as e:
                print(f"Error fetching from Task Dispatcher: {e}")

            time.sleep(self.polling_interval)

    def process_result(self, result: dict):
        detections = result.get("results", {})
        stream_id = result.get("stream_id", "unknown")
        image = result.get("image")
        #if any detection in any frame_type has confidence > 0.8, trigger next service
        for frame_type, data in detections.items():
            for det in data:
                
                if det.get("confidence", 0) > 0.8:
                    print("Found above")
                    # self.trigger_sceneunderstanding(stream_id, frame_type, det, image)
                    self.trigger_test(stream_id, frame_type, det, image)

    def trigger_sceneunderstanding(self, stream_id, frame_type, detection, image):
        try:
            payload = {
                "stream_id": stream_id,
                "frame_type": frame_type,
                "detection": detection,
                "image": image
            }
            response = requests.post(self.trigger_url, json=payload)
            print(f"Triggered next service: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error triggering next service: {e}")

    def trigger_test(self, stream_id, frame_type, detection, image):
        payload = {
            "stream_id": stream_id,
            "frame_type": frame_type,
            "detection": detection,
            "image": image
        }
        print(stream_id, " : ",frame_type, " : ", detection)