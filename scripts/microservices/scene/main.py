from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import cv2
import uvicorn
import base64
import numpy as np

from logger import logger
from refactored_scene_understanding import create_scene_understanding, OllamaSceneUnderstanding

VIDEO_STREAM_SERVICE_URL = "http://localhost:8000/update_priority/"
ANOMALY_URL = "http://localhost:8005/analyze/"

usr_prompt = (
    "You are a scene understanding model tasked with describing images in a sequence. "
    "Please analyze the following images and provide **detailed and precise descriptions**. "
    "For each image, include the following information in your response: "
    "- **People**: The number of individuals in the scene and a description of their actions. "
    "- **Clothing**: Detailed description of clothing, including type (e.g., uniforms, casual wear), color, and any distinct features. "
    "- **Objects**: List and describe objects in the scene (e.g., vehicles, furniture, accessories), including their condition and significance. "
    "- **Backgroteractiund**: Provide details about the environment or setting, such as location (e.g., street, office, park), time of day (e.g., daytime, nighttime), and any notable background elements. "
    "- **Inons**: If applicable, describe how people and objects interact in the scene. "
    "Ensure that every detail is based solely on visual information, and avoid making assumptions or adding extra details. "
    "After describing each image, please provide a **summary of how the scene progresses** from one frame to the next, highlighting any changes in action, environment, or key elements."
)
system_prompt = (
    "You are a security analyst."
)

app = FastAPI()
scene_model: OllamaSceneUnderstanding = create_scene_understanding('llama3', prompt=system_prompt)
session = requests.Session()

class ProcessEventRequest(BaseModel):
    stream_id: str
    object_frame: list
    activity_frame: list
    emotion_frame: list
    image: str

def decode_image(encoded_frame):
    decoded = base64.b64decode(encoded_frame)
    image_arr = np.frombuffer(decoded, dtype=np.uint8)
    image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
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

def call_anomaly_microservice(result:str, stream_id:str, combined_det: str):
    logger.info("Testing: Called anomaly model")
    try:
        response = session.post(
            ANOMALY_URL,
            json={
                "stream_id": stream_id,
                "detections": combined_det,
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
    _, obj = scene_model._process_features("", req.object_frame)
    _, act = scene_model._process_features("", req.activity_frame)
    _, emo = scene_model._process_features("", req.emotion_frame)
    logger.debug(f"obj: {obj}")
    logger.debug(f"act: {act}")
    logger.debug(f"emo: {emo}")
    combined = "\n".join([obj,act,emo])
    try:
        image = decode_image(req.image)
    except Exception as e:
        logger.error(f"Exception occured:  {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

    result: str = scene_model.analyze_scene(image,usr_prompt)

    logger.info(f"Scene Caption: {result}")
    # ------- tool call ----------- # 
    # action = result.get("action")
    # if action == "update_priority":
    #     call_videostreamhandler_microservice(result, stream_id)
    # elif action == "trigger_anomaly":
    #     call_anomaly_microservice(result,stream_id,frame_type,detection)

    #  Testing
    
    out = call_anomaly_microservice(result,stream_id,combined)
    logger.info(out)
    return {
        "status": "no_action",
        "caption": result,
    }

@app.get("/status")
def status():
    return {"service": "Scene Understanding", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
