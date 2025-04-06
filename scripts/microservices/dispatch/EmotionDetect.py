import cv2
from ultralytics import YOLO
from BaseModel import BaseModel
from Detections import Detections

class EmotionDetect(BaseModel):
    def __init__(self, emotion_model_path, face_cascade_path):
        super().__init__()
        self.model = YOLO(emotion_model_path)
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect(self, image):
        detections = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(3, 3))

        for (x, y, w, h) in faces:
            face_region = image[y:y+h, x:x+w]
            results = self.model.predict(face_region, verbose=False)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]

                    x1 += x
                    x2 += x
                    y1 += y
                    y2 += y

                    det = Detections(label, confidence, [x1, y1, x2, y2])
                    detections.append(det)

        self.out = {"detections": detections}
        return self.out
