import torch
import io
import time
import os
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
    
    def _process_features(self, scene: str, objects: list[dict])->tuple[str,str]:
        object_description = "\n".join(
            f"- {obj.get('label', 'unknown')} "
            f"(Confidence: {obj.get('confidence', 0)*100:.1f}%) "
            f"at {obj.get('bbox', [])}"
            for obj in objects['detections']
        )
        return scene,object_description
    
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
    
    # Class-level caption definitions
    POTENTIAL_CAPTIONS = [
        "a photograph of a peaceful scene",
        "a crime scene with a weapon",
        "people engaged in normal activities",
        "a busy scene with many people",
        "a violent confrontation or fight",
    ]
    
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
        best_caption, confidence = self.rank_captions(self.POTENTIAL_CAPTIONS)
        self.out = best_caption
        return self.out

    def rank_captions(self, potential_captions=None):
        """Rank captions based on similarity to image"""
        if potential_captions is None:
            potential_captions = self.POTENTIAL_CAPTIONS
            
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
    
    # Class-level prompt definition
    STARTING_PROMPT = (
        "Analyze the image carefully. If you notice any elements related to crime, "
        "such as a crime scene, weapon, gun, knife, violence, fight, or blood, "
        "please mention it in the caption. If none of these are present, simply describe "
        "the scene without reference to these terms."
    )
    
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def analyze_scene(self, image, text=None):
        """Generate caption for an image with optional text prompt"""
        image_data = self.load_image(image)
        return self.generate_caption(image_data, text)
    
    def generate_caption(self, image, text=None):
        """Generate caption for the provided image"""
        text_prompt = text if text else self.STARTING_PROMPT
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


class OllamaSceneUnderstanding(BaseSceneUnderstanding):
    """Scene understanding using Ollama with various vision-language models"""
    
    # Class-level prompt definitions
    LLAVA_PROMPT = (
        "Analyze the image carefully and provide a detailed description of the scene. "
        "If there are any elements related to crime (e.g., a weapon, gun, knife, violence, fight, blood, suspicious activity, etc), "
        "mention them. ANSWER IN 50 WORDS"
    )
    
    LLAMA3_PROMPT = (
        "Analyze the image carefully and provide a detailed description of the scene. "
        "If there are any elements related to crime (e.g., a weapon, gun, knife, violence, fight, blood, or suspicious activity), "
        "mention them and explicitly state: 'Crime detected: YES'. "
        "If no crime-related elements are present, explicitly state: 'Crime detected: NO'. "
        "Ensure your response follows this exact format."
    )
    
    # Model options
    MODEL_OPTIONS = {
        "num_ctx": 2048,
        "num_gpu_layers": 20,
        "temperature": 0
    }
    
    def __init__(self, model="llava:7b"):
        super().__init__()
        self.model = model
        
        # Select appropriate prompt based on model type
        if "llama3" in model.lower():
            self.prompt = self.LLAMA3_PROMPT
        else:
            self.prompt = self.LLAVA_PROMPT

    def analyze_scene(self, image_path):
        """Process the image with Ollama model and return a scene description"""
        image = self.load_image(image_path)
        
        # Convert image to bytes for Ollama processing
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        
        # Generate response using ollama.generate for consistency across all models
        t0 = time.time()
        response = ollama.generate(
            model=self.model,
            options=self.MODEL_OPTIONS,
            prompt=self.prompt,
            images=[image_bytes]
        )
        t1 = time.time()
        print(f"Model {self.model} took: {t1-t0:.2f} sec")
        
        self.out = response['response']
        return self.out


class DynamicSceneUnderstanding(OllamaSceneUnderstanding):
    """Extended Ollama scene understanding for analyzing sequences of images"""
    
    # Custom prompt for multi-frame analysis as a class attribute
    MULTI_FRAME_PROMPT = (
        "You are a scene understanding model tasked with describing images in a sequence. "
        "Please analyze the following images and provide **detailed and precise descriptions**. "
        "For each image, include the following information in your response: "
        "- **People**: The number of individuals in the scene and a description of their actions. "
        "- **Clothing**: Detailed description of clothing, including type (e.g., uniforms, casual wear), color, and any distinct features. "
        "- **Objects**: List and describe objects in the scene (e.g., vehicles, furniture, accessories), including their condition and significance. "
        "- **Background**: Provide details about the environment or setting, such as location (e.g., street, office, park), time of day (e.g., daytime, nighttime), and any notable background elements. "
        "- **Interactions**: If applicable, describe how people and objects interact in the scene. "
        "Ensure that every detail is based solely on visual information, and avoid making assumptions or adding extra details. "
        "After describing each image, please provide a **summary of how the scene progresses** from one frame to the next, highlighting any changes in action, environment, or key elements."
    )
    
    SYSTEM_MESSAGE = "You are analyzing images to understand dynamic scenes."

    def __init__(self, model="llava:13b"):
        super().__init__(model)
        
    def load_images_from_folder(self, folder_path, max_images=5):
        """Loads up to max_images from the given folder"""
        images = []
        valid_extensions = (".jpg", ".jpeg", ".png")

        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])
        for image_file in image_files[:max_images]:
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path).convert("RGB")
            images.append(img)

        return images
        
    def analyze_scene_sequence(self, frames_or_folder):
        """Analyze multiple frames and describe the scene changes"""
        # Handle folder path or list of frames
        if isinstance(frames_or_folder, str) and os.path.isdir(frames_or_folder):
            frames = self.load_images_from_folder(frames_or_folder)
        else:
            frames = frames_or_folder

        # Convert images to bytes
        image_bytes_list = []
        for image in frames:
            img_bytes = io.BytesIO()
            image = self.load_image(image)  # Ensure it's a PIL image
            image.save(img_bytes, format="JPEG")
            image_bytes_list.append(img_bytes.getvalue())
            
        # For multi-frame analysis, we need to use ollama.chat which supports multiple images
        # This is an exception to the generate-only rule, but necessary for this functionality
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": self.MULTI_FRAME_PROMPT, "images": image_bytes_list}
        ])

        self.out = response['message']['content']
        return self.out


# Factory function to create scene understanding model instances
def create_scene_understanding(model_type, **kwargs):
    """
    Factory function to create the appropriate scene understanding model
    
    Args:
        model_type: String identifier for model type (clip, blip, ollama, dynamic)
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        An instance of a scene understanding model
    """
    if model_type.lower() == "clip":
        return CLIPSceneUnderstanding(**kwargs)
    elif model_type.lower() == "blip":
        return BLIPSceneUnderstanding(**kwargs)
    elif model_type.lower() == "dynamic":
        return DynamicSceneUnderstanding(**kwargs)
    else:
        # Handle Ollama models
        if model_type.lower() in ["ollama", "llava", "llama3"]:
            # Use default model name if not specified
            model_name = kwargs.get("model")
            if not model_name:
                if model_type.lower() == "llama3":
                    model_name = "llama3.2-vision:latest"
                else:
                    model_name = "llava:7b"
            return OllamaSceneUnderstanding(model=model_name)
        elif ":" in model_type:  # Model specified as "llava:13b" or similar
            return OllamaSceneUnderstanding(model=model_type)
            
        raise ValueError(f"Unknown model type: {model_type}")