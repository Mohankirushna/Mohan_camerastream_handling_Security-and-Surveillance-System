import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64

from Detections import Detections
from BufferItem import BufferItem

from TaskDispatcher import TaskDispatcher
from YOLODetector import YOLODetector
from EmotionDetect import EmotionDetect

models = {
    "object_frame": YOLODetector("models/yolov8n.pt"),
    "emotion_frame": EmotionDetect(
        "models/yolov8_emotion_detection.pt",
        "models/haarcascade_frontalface_default.xml"
    ),
    "activity_frame": YOLODetector("models/yolov8_human_activity_detection.pt"),
}

task_dispatcher = TaskDispatcher(models, pull_url = "")

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


images = {
    "object_frame" : img,
    "emotion_frame" : img,
    "activity_frame" : img,
}

results = task_dispatcher.dispatch(images)
_, encoded_frame = cv2.imencode('.jpg', img)
encoded_b64 = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')
buff = BufferItem("stream1", results, encoded_b64)

annotated_img = buff.to_annotated_image()
print(buff.__dict__())
plt.imshow(annotated_img)
plt.show()

recreated_res = {
    'object_frame': [
        {
            'confidence': 0.99, 
            'label': 'person', 
            'bbox': [100, 40, 150, 110]
        }
    ], 
    'emotion_frame': [], 
    'activity_frame': []
}

recreated_buff = BufferItem("stream1", recreated_res, encoded_b64)
annotated_img = recreated_buff.to_annotated_image()
plt.imshow(annotated_img)
plt.show()