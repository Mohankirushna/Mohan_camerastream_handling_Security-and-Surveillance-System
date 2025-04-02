#!pip install ultralytics

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scripts.obj_det.BaseModel import BaseModel

class YOLODetector(BaseModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = YOLO(model_path)
        print(self.model.info())

    def detect(self, image):
        results = self.model.predict(image, verbose=False)
        return self._process_results(results[0])

    def _process_results(self, result):
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })
        self.out = {"detections": detections}
        return self.out

    def visualize_results(self, image_path, results):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['label']} {detection['confidence']:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()



