from ultralytics import YOLO
from BaseModel import BaseModel
from Detections import Detections
from logger import logger

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
            if confidence > 0.5:
                det = Detections(label, confidence, [x1, y1, x2, y2])
                detections.append(det)
            else:
                logger.info(f"low confidence <=0.5 : {label} {confidence} {[x1, y1, x2, y2]}")
        self.out = {"detections": detections}
        return self.out



