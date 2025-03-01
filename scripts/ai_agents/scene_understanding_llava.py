import ollama
from PIL import Image
import io
import numpy as np
import time

class OllavaSceneUnderstanding:
    def __init__(self, model):
        self.model = model  # Ensure the LLaVA model is downloaded

    def load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def detect(self, image):
        return self.analyze_scene(image)

    def analyze_scene(self, image_path):
        if isinstance(image_path, str):
            image = self.load_image(image_path)
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path).convert("RGB")
        else:
            image = image_path

        # Convert image to bytes for Ollama processing
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()

        # Enhanced prompt to ensure a structured YES/NO response
        prompt = (
            "Analyze the image carefully and provide a detailed description of the scene. "
            "If there are any elements related to crime (e.g., a weapon, gun, knife, violence, fight, blood, suspicious activity, etc), "
            "mention them. ANSWER IN 50 WORDS"
        )

        # Generate response
        t0 = time.time()
        response = ollama.generate(
            model=self.model, 
            options={
                "num_ctx": 2048,  # Reduce context window
                "num_gpu_layers": 20,    # Allocate more layers to GPU
                "temperature": 0  # Disable randomness
            },
            prompt = prompt,
            images = [image_bytes]
        )
        t1 = time.time()
        print("model took:",t1-t0, 'sec')

        # Extract text response
        self.out = response['response']
        return self.out
