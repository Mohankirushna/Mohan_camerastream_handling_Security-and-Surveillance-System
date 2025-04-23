from fastapi import FastAPI, HTTPException
import uvicorn

from FrameProcessor import FrameProcessor
from PreprocessingManager import PreprocessingManager
from logger import logger

VIDEO_STREAM_SERVICE_URL = "http://localhost:8000/get_frame/"

app = FastAPI()

@app.get("/get_processed_frame/")
def get_processed_frame():
    frame_data = preprocessing_manager.get_latest_frame()
    if frame_data:
        return frame_data
    logger.error("Data Could not be processed")
    raise HTTPException(status_code=404, detail="No processed frame available")

frame_processor = FrameProcessor()
preprocessing_manager = PreprocessingManager(frame_processor, pull_url=VIDEO_STREAM_SERVICE_URL,buffer_size=100)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
