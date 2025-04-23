from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from refactored_scene_understanding import create_scene_understanding

import requests
from logger import logger

# Endpoint that may be triggered by anomaly detection
VIDEO_STREAM_SERVICE_URL = "http://videostream-service:8000/update_priority/"

system_prompt = (
    "You are a security analyst."
)
usr_prompt = (
    "Provide assessment in EXACTLY this format:\n"
    "Security Concern: <concern>\n"
    "Reason: <reason>\n"
    "Level: <low|medium|high>\n"
    "Confidence: <percentage>%\n\n"
    "Base analysis on these inputs:"
)

app = FastAPI()
anomaly_model = create_scene_understanding(model_type='llama3.1:8b', prompt=system_prompt)
session = requests.Session()

class ProcessEventRequest(BaseModel):
    stream_id: str
    detections: str
    caption: str

def call_videostreamhandling_microservice(result,req):
    #TODO: Fix for actual priority. needs to be in sync with the base model
    priority = result.get("priority", 3)
    try:
        response = session.post(
            VIDEO_STREAM_SERVICE_URL,
            json={"stream_id": req.stream_id, "priority": priority}
        )
        return {
            "status": "priority_updated",
            "caption": req.caption,
            "decision": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update priority: {e}")

@app.post("/analyze/")
async def analyze(req: ProcessEventRequest):
    try:
        result = anomaly_model.analyze_scene(None, usr_prompt + f"Detections: {req.detections}\nSCENE: {req.caption}")
        # result: str = anomaly_model.analyze(req.caption, req.detection)
        logger.info(result)
        # action = result.get("action")
        # if action == "update_priority":
        #     call_videostreamhandling_microservice(result, req)
        return {
            "status": "no_action",
            "caption": req.caption,
            "decision": result
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

@app.get("/status")
def status():
    return {"service": "Anomaly Detection", "status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
