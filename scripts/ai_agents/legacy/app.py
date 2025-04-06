from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from scripts.ai_agents.legacy.scene_understanding_blip_salesforce import SceneUnderstanding
from io import BytesIO

app = FastAPI(title="Scene Understanding Microservice",
              description="Receives an image, generates a caption and classifies crime.",
              version="1.0")

# Initialize model on startup instead of at module load
@app.on_event("startup")
async def startup_event():
    # Log TASK ID: SU-7 for tracking purposes.
    print("Starting Scene Understanding Microservice - TASK ID: SU-7")
    app.state.scene_understander = SceneUnderstanding()


@app.post("/process")
async def process_image(file: UploadFile = File(None), image_url: str = Form(None), prompt: str = Form(None)):
    """
    Receives an image either as an upload or via image_url.
    An optional prompt can be provided for caption generation.
    Returns the generated caption and crime classification.
    """
    try:
        scene_understander = app.state.scene_understander

        if file:
            contents = await file.read()
            image = scene_understander.load_image(image_data=BytesIO(contents))
        elif image_url:
            image = scene_understander.load_image(url=image_url)
        else:
            raise HTTPException(status_code=400, detail="No image file or image_url provided.")
        
        caption = scene_understander.generate_caption(image, text=prompt)
        classification = scene_understander.classify_crime(caption)
        
        return JSONResponse(content={"caption": caption, "classification": classification})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Scene Understanding Microservice is running (TASK ID: SU-7)."}


@app.post("/save_model")
def save_model(save_path: str = Form("/path/to/models")):
    try:
        scene_understander = app.state.scene_understander
        scene_understander.save_model(save_path)
        return JSONResponse(content={"message": f"Model saved to {save_path}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # When running under Windows, this __main__ guard prevents issues with spawning processes.
    uvicorn.run(app, host="0.0.0.0", port=8000)


#http://localhost:8000/docs