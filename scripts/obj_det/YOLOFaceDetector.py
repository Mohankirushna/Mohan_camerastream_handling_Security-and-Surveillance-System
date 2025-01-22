import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

class YOLOFaceDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)

    def detect_faces(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
        results = self.model(image_path)
        return results

    @staticmethod
    def visualize_results(image_path, results):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = f"{results[0].names[int(box.cls[0])]} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
