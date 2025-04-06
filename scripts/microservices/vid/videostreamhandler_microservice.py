from fastapi import FastAPI, HTTPException
import uvicorn

from VideoStreamHandler import VideoStreamHandler

app = FastAPI()
handler = VideoStreamHandler(5)

@app.post("/add_stream/")
def add_stream(stream_id: int, source: str, priority: int):
    try:
        handler.add_video_stream(stream_id, source, priority)
        return {"success": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_frame/")
def get_frame():
    return handler.get_frame()

@app.put("/update_priority/")
def update_priority(stream_id: int, new_priority: int):
    return handler.update_priority(stream_id, new_priority)

if __name__ == "__main__":
    source = "video4.mp4"
    # for now
    handler.add_video_stream('stream2', source, 1)
    handler.add_video_stream('stream1', source, 1)

    handler.start_all()

    uvicorn.run(app, host="0.0.0.0", port=8000)