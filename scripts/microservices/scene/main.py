from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import cv2
import uvicorn

from logger import logger
from refactored_scene_understanding import create_scene_understanding

VIDEO_STREAM_SERVICE_URL = "http://localhost:8000/update_priority/"
ANOMALY_URL = "http://localhost:8005/analyze/"

app = FastAPI()
scene_model = create_scene_understanding('llama3')
session = requests.Session()

class ProcessEventRequest(BaseModel):
    stream_id: str
    frame_type: str
    detection: dict
    image: str

def decode_image(encoded_frame):
    image = cv2.imdecode(bytearray(encoded_frame), cv2.IMREAD_COLOR)
    return image

def call_videostreamhandler_microservice(result, stream_id):
    priority = result.get("priority")
    try:
        response = session.post(
            VIDEO_STREAM_SERVICE_URL,
            json={"stream_id": stream_id, "priority": priority}
        )
        return {
            "status": "priority_updated",
            "caption": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update priority: {e}")

def call_anomaly_microservice(result, stream_id, frame_type, detection):
    logger.info("Testing: Called anomaly model")
    try:
        response = session.post(
            ANOMALY_URL,
            json={
                "stream_id": stream_id,
                "frame_type": frame_type,
                "detection": detection,
                "caption": result
            }
        )
        return {
            "status": "anomaly_triggered",
            "caption": result,
        }
    except Exception as e:
        logger.error(f"Failed to trigger anomly detection model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger anomaly detection: {e}")

@app.post("/process_event")
async def process_event(req: ProcessEventRequest):
    stream_id = req.stream_id
    frame_type = req.frame_type
    detection = req.detection

    try:
        image = decode_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")

    scene_model.set_image(image)
    result: str = scene_model.detect(image)

    logger.info(f"Scene Caption: {result}")
    # ------- tool call ----------- # 
    # action = result.get("action")
    # if action == "update_priority":
    #     call_videostreamhandler_microservice(result, stream_id)
    # elif action == "trigger_anomaly":
    #     call_anomaly_microservice(result,stream_id,frame_type,detection)

    #  Testing
    
    out = call_anomaly_microservice(result,stream_id,frame_type,detection)
    print(out)

    return {
        "status": "no_action",
        "caption": result,
    }

@app.get("/status")
def status():
    return {"service": "Scene Understanding", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
