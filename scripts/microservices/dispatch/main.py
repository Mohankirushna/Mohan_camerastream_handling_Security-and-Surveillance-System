
from fastapi import FastAPI, HTTPException
import uvicorn

from TaskDispatcher import TaskDispatcher
from YOLODetector import YOLODetector
from EmotionDetect import EmotionDetect
from logger import logger

PREPROCESSING_SERVICE_URL = "http://localhost:8001/get_processed_frame/"

# Initialize models dynamically
models = {
    "object_frame": YOLODetector("models/yolov8n.pt"),
    "emotion_frame": EmotionDetect(
        "models/yolov8_human_activity_detection.pt",
        "models/haarcascade_frontalface_default.xml"
    ),
    "activity_frame": YOLODetector("models/yolov8_human_activity_detection.pt"),
}

task_dispatcher = TaskDispatcher(models, pull_url = PREPROCESSING_SERVICE_URL)

app = FastAPI()

@app.get("/get_results/")
def get_results():
    result = task_dispatcher.get_latest_result()
    if result:
        logger.info("Processed Results from all models")
        logger.info(str(result.__dict__()))
        return result.__dict__()
    raise HTTPException(status_code=404, detail="No results available")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
