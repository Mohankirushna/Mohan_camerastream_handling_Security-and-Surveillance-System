import requests
import numpy as np
import cv2
import base64
import json

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

_, encoded_frame = cv2.imencode('.jpg', img)
encoded_b64 = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')

payload = {
    "stream_id": "stream1",
    "object_frame": [
        {"confidence":0.89, "label": "person", "bbox": [100, 0, 150, 150]}
    ],
    "activity_frame": [{"confidence":0.69, "label": "walking", "bbox": [120, 0, 170, 100]}],
    "emotion_frame": [],
    "image": encoded_b64
}
# print(payload)
session = requests.Session()
resp = session.post(
    "http://localhost:8004/process_event",
    json=payload
)
print(resp.json())