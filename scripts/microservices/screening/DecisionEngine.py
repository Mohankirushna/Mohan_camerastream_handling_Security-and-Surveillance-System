import time
import requests
import threading
from logger import logger

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
                logger.error(f"Error fetching from Task Dispatcher: {e}")

            time.sleep(self.polling_interval)

    def process_result(self, result: dict):
        detections = result.get("results", {})
        stream_id = result.get("stream_id", "unknown")
        image = result.get("image")
        payload = {
            "stream_id": stream_id,
            "image": image,
            "object_frame": [],
            "activity_frame": [],
            "emotion_frame": []
        }
        # if any detection in any frame_type has confidence > 0.8, trigger next service
        trigger = False 
        for frame_type, data in detections.items():
            for det in data:
                if self.trigger_condition(frame_type, det):
                    if frame_type not in payload:
                        payload[frame_type] = []
                    payload[frame_type].append(det)
                    trigger = True
        if trigger:
            # self.trigger_test(payload)
            self.trigger_sceneunderstanding(payload)
    
    def trigger_condition(self, frame_type, det):
        return det.get("confidence", 0) > 0.8

    def trigger_sceneunderstanding(self, payload):
        try:
            response = requests.post(self.trigger_url, json=payload)
            logger.info(f"Triggered next service (resp code): {response.status_code}")
            if response.status_code != 200:
                logger.error("Error triggering the next service: Bad Error Code")
        except requests.RequestException as e:
            logger.error(f"Error triggering next service: {e}")

    def trigger_test(self, payload:dict):
        logger.info("Testing triggered")
        toprint_payload = {k:v for k,v in payload.items() if k != 'image'}
        logger.info(toprint_payload)