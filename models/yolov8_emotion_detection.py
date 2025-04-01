import cv2
import ultralytics
from ultralytics import YOLO

class EmotionDetect:
    def __init__(self, emotion_model_path, face_cascade_path):
        self.emotion_model = YOLO(emotion_model_path)
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect(self, source):
        image = cv2.imread(source)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(3, 3))
        
        for (x, y, w, h) in faces:
            face_region = image[y:y+h, x:x+w]
            results = self.emotion_model(face_region)
            
            for result in results:
                classes = result.names
                cls = result.boxes.cls
                conf = result.boxes.conf
                detections = result.boxes.xyxy

                for pos, detection in enumerate(detections):
                    xmin, ymin, xmax, ymax = detection
                    xmin_full = int(xmin) + x
                    ymin_full = int(ymin) + y
                    xmax_full = int(xmax) + x
                    ymax_full = int(ymax) + y
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(image, (xmin_full, ymin_full), (xmax_full, ymax_full), color, 2)
                    cv2.putText(image, label, (xmin_full, ymin_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        cv2.imshow("Emotion Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_model_path = 'yoloEmotion.pt'
    face_cascade_path = 'haarcascade_frontalface_default.xml' 
    source = 'image.jpeg'
    detector = EmotionDetect(emotion_model_path, face_cascade_path)
    detector.detect(source)
