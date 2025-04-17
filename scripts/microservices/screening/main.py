from fastapi import FastAPI
import uvicorn

from DecisionEngine import DecisionEngine

TASK_DISPATCHER_URL = "http://localhost:8002/get_results/"
SCENE_UNDERSTANDING_URL = "http://localhost:8004/process_event"

app = FastAPI()

# Initialize and start engine
decision_engine = DecisionEngine(
    pull_url = TASK_DISPATCHER_URL, 
    trigger_url = SCENE_UNDERSTANDING_URL,
    polling_interval=0.01
)
decision_engine.start()

@app.get("/status")
def status():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
