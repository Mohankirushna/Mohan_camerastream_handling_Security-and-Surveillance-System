# File: scene_understanding.py
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneUnderstanding:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.starting_prompt = (
            "Analyze the image carefully. If you notice any elements related to crime, "
            "such as a crime scene, weapon, gun, knife, violence, fight, or blood, "
            "please mention it in the caption. If none of these are present, simply describe "
            "the scene without reference to these terms."
        )
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
        ).to(self.device)
    
    def detect(self, img):
        self.raw_image = Image.fromarray(img).convert("RGB")
        self.generate_caption()
    
    def set_image(self, img: Image):
        self.raw_image = img

    def generate_caption(self, text=None):
        if text != None:
            inputs = self.processor(self.raw_image, text, return_tensors="pt", max_length=200, truncation=True)
    def load_image(self, image_data=None, url=None):
        if image_data:
            return Image.open(image_data).convert("RGB")
        elif url:
            return Image.open(requests.get(url, stream=True).raw).convert("RGB")
        else:
            raise ValueError("No image data or URL provided")
    
    def generate_caption(self, image, text=None):
        inputs = self.processor(image, text, return_tensors="pt") if text \
            else self.processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.out = self.model.generate(**inputs)
        self.out = self.processor.decode(self.out[0], skip_special_tokens=True)
        return self.out
    
    def classify_crime(self, caption):
        crime_keywords = ["crime", "weapon", "gun", "knife", "violence", "fight", "blood", "pistol"]
        if any(keyword in caption.lower() for keyword in crime_keywords):
            return "Crime detected"
        return "No crime"

    def save_model(self, save_dir="saved_blip"):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)