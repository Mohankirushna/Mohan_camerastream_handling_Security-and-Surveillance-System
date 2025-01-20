import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneUnderstanding:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
        ).to("cuda")
        self.img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
        self.raw_image = Image.open(requests.get(self.img_url, stream=True).raw).convert('RGB')

    def generate_caption(self, text=None):
        if text:
            inputs = self.processor(self.raw_image, text, return_tensors="pt").to("cuda", torch.float16)
        else:
            inputs = self.processor(self.raw_image, return_tensors="pt").to("cuda", torch.float16)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def classify_crime(self, caption):
        crime_keywords = ["crime", "weapon", "gun", "knife", "violence", "fight", "blood"]
        if any(keyword in caption.lower() for keyword in crime_keywords):
            return "Crime detected"
        return "No crime"

    def save_model(self, save_dir="saved_blip"):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)

if __name__ == "__main__":
    scene_understanding = SceneUnderstanding()
    caption = scene_understanding.generate_caption("a photography of")
    print(caption)
    print(scene_understanding.classify_crime(caption))
    caption = scene_understanding.generate_caption()
    print(caption)
    print(scene_understanding.classify_crime(caption))

    # Integrate save here
    scene_understanding.save_model("/path/to/models")