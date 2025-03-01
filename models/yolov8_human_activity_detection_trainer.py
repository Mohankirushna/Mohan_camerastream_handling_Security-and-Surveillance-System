import os
import torch
import kagglehub
from ultralytics import YOLO

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Define dataset name on Kaggle
KAGGLE_DATASET = "sanjay0614/human-activity-detection"

# Define dataset save path
DATASET_PATH = "<SPECIFY PATH>"

# Function to download dataset using kagglehub
def download_dataset():
    print("Downloading dataset from Kaggle...")
    kagglehub.dataset_download(KAGGLE_DATASET, path=DATASET_PATH)
    print(f"Dataset downloaded to: {DATASET_PATH}")

# Download dataset
download_dataset()

# Define paths
DATA_YAML_PATH = os.path.join(DATASET_PATH, "human-activity-detection-2-2", "data.yaml")

# Define model save path
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "yolov8_human_activity_detection.pt")

# Ensure the models directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano

# Train the model
print("Training YOLOv8 on human activity detection dataset...")
model.train(
    data=DATA_YAML_PATH,  # Path to dataset YAML file
    epochs=10,
    batch=16,
    imgsz=640,
    device=device
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved at: {MODEL_SAVE_PATH}")
