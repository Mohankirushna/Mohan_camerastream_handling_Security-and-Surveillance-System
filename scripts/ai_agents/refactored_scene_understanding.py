import torch
import io
import time
import numpy as np
import requests
import ollama
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

class BaseSceneUnderstanding:
    """Base class for all scene understanding models"""
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.out = None

    def set_image(self, img):
        """Set the image for processing"""
        self.raw_image = img

    def load_image(self, image_path=None, url=None, image_data=None):
        """Load an image from path, URL, or raw data"""
        if image_path is not None:
            if isinstance(image_path, str):
                return Image.open(image_path).convert("RGB")
            elif isinstance(image_path, np.ndarray):
                return Image.fromarray(image_path).convert("RGB")
            else:
                return image_path
        elif url is not None:
            return Image.open(requests.get(url, stream=True).raw).convert("RGB")
        elif image_data is not None:
            return Image.open(image_data).convert("RGB")
        return None
    
    def detect(self, image):
        """Common detection interface for integration with other components"""
        return self.analyze_scene(image)
    
    def analyze_scene(self, image):
        """To be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement analyze_scene method")
        
    def classify_crime(self, caption):
        """Classify if the caption contains crime-related content"""
        crime_keywords = ["crime", "weapon", "gun", "knife", "violence", "fight", "blood", "pistol"]
        if any(keyword in caption.lower() for keyword in crime_keywords):
            return "Crime detected"
        return "No crime"
    
    def save_model(self, save_dir):
        """Save model to disk - to be implemented by subclasses with saveable models"""
        raise NotImplementedError("This model doesn't support saving")


class CLIPSceneUnderstanding(BaseSceneUnderstanding):
    """Scene understanding using OpenAI's CLIP model for ranking captions"""
    
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        try:
            self.model.to(self.device)
        except Exception:
            print("Failed to load CLIP model to GPU")

    def analyze_scene(self, image):
        """Analyze scene using predefined captions"""
        self.set_image(self.load_image(image))
        text_prompts = [
            "a photograph of a peaceful scene",
            "a crime scene with a weapon",
            "people engaged in normal activities",
            "a busy scene with many people",
            "a violent confrontation or fight",
        ]
        best_caption, confidence = self.rank_captions(text_prompts)
        self.out = best_caption
        return self.out

    def rank_captions(self, potential_captions):
        """Rank captions based on similarity to image"""
        inputs = self.processor(
            text=potential_captions, images=self.raw_image, return_tensors="pt", padding=True
        )
        if self.model.device.type == "cuda":
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        best_idx = probs.argmax(dim=1).item()
        return potential_captions[best_idx], probs[0, best_idx].item()
    
    def save_model(self, save_dir):
        """Save CLIP model and processor to disk"""
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)


class BLIPSceneUnderstanding(BaseSceneUnderstanding):
    """Scene understanding using Salesforce BLIP model for caption generation"""
    
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.starting_prompt = (
            "Analyze the image carefully. If you notice any elements related to crime, "
            "such as a crime scene, weapon, gun, knife, violence, fight, or blood, "
            "please mention it in the caption."
        )

    def analyze_scene(self, image, text=None):
        """Generate caption for an image with optional text prompt"""
        image_data = self.load_image(image)
        return self.generate_caption(image_data, text)
    
    def generate_caption(self, image, text=None):
        """Generate caption for the provided image"""
        text_prompt = text if text else self.starting_prompt
        inputs = self.processor(image, text_prompt, return_tensors="pt") if text_prompt \
            else self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        out = self.model.generate(**inputs)
        self.out = self.processor.decode(out[0], skip_special_tokens=True)
        return self.out
    
    def save_model(self, save_dir):
        """Save BLIP model and processor to disk"""
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)


class LLaVASceneUnderstanding(BaseSceneUnderstanding):
    """Scene understanding using Ollama with LLaVA model"""
    
    def __init__(self, model="llava:7b"):
        super().__init__()
        self.model = model

    def analyze_scene(self, image_path):
        """Process the image with Ollama LLaVA and return a scene description"""
        image = self.load_image(image_path)
        
        # Convert image to bytes for Ollama processing
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        
        # Prompt for scene analysis
        prompt = (
            "Analyze the image carefully and provide a detailed description of the scene. "
            "If there are any elements related to crime (e.g., a weapon, gun, knife, violence, fight, blood, suspicious activity, etc), "
            "mention them. ANSWER IN 50 WORDS"
        )
        
        # Generate response using ollama.generate
        t0 = time.time()
        response = ollama.generate(
            model=self.model, 
            options={
                "num_ctx": 2048,
                "num_gpu_layers": 20,
                "temperature": 0
            },
            prompt=prompt,
            images=[image_bytes]
        )
        t1 = time.time()
        print("model took:", t1-t0, 'sec')
        
        self.out = response['response']
        return self.out


class Llama3VisionSceneUnderstanding(BaseSceneUnderstanding):
    """Scene understanding using Ollama with Llama 3.2 Vision model"""
    
    def __init__(self):
        super().__init__()
        self.model = "llama3.2-vision:latest"

    def analyze_scene(self, image_path):
        """Process the image with Llama 3.2 and return a scene description"""
        image = self.load_image(image_path)
        
        # Convert image to bytes for Ollama processing
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        
        # Enhanced prompt to ensure a structured YES/NO response
        prompt = (
            "Analyze the image carefully and provide a detailed description of the scene. "
            "If there are any elements related to crime (e.g., a weapon, gun, knife, violence, fight, blood, or suspicious activity), "
            "mention them and explicitly state: 'Crime detected: YES'. "
            "If no crime-related elements are present, explicitly state: 'Crime detected: NO'. "
            "Ensure your response follows this exact format."
        )
        
        # Generate response using ollama.chat (different from LLaVA's generate)
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are an AI security analyst reviewing surveillance images for potential crime detection."},
            {"role": "user", "content": prompt, "images": [image_bytes]}
        ])
        
        self.out = response['message']['content']
        return self.out


# Factory function to create scene understanding model instances
def create_scene_understanding(model_type, **kwargs):
    """
    Factory function to create the appropriate scene understanding model
    
    Args:
        model_type: String identifier for model type (clip, blip, llava, llama3)
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        An instance of a scene understanding model
    """
    if model_type.lower() == "clip":
        return CLIPSceneUnderstanding(**kwargs)
    elif model_type.lower() == "blip":
        return BLIPSceneUnderstanding(**kwargs)
    elif model_type.lower() == "llava":
        return LLaVASceneUnderstanding(**kwargs)
    elif model_type.lower() == "llama3":
        return Llama3VisionSceneUnderstanding(**kwargs)
    else:
        # Default to LLaVA when given a specific model name
        if "llava" in model_type.lower():
            return LLaVASceneUnderstanding(model=model_type, **kwargs)
        elif "llama3" in model_type.lower() or "llama-3" in model_type.lower():
            return Llama3VisionSceneUnderstanding()
        raise ValueError(f"Unknown model type: {model_type}")