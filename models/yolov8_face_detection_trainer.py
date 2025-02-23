from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

# Define model save path
save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_save_path = os.path.join(save_dir, "yolov8_face_detection.pt")

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model on your face detection dataset
model.train(
    data='face.yaml',  # Dataset configuration file
    epochs=5,
    imgsz=640,
    batch=16,
    workers=4,
    name='face_detection',
    verbose=True,
    project='runs/detect',
    device='cuda'
)

# Save the trained model
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")
