from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneUnderstanding:
    def __init__(self, device='cpu'):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.device = device
        if device == 'cuda':
            self.model.to("cuda")
    
    def set_image(self, img: Image):
        self.raw_image = img

    def generate_caption(self, text=None):
        if text != None:
            inputs = self.processor(self.raw_image, text, return_tensors="pt", max_length=200, truncation=True)
        else:
            inputs = self.processor(self.raw_image, return_tensors="pt", max_length=200, truncation=True)
        if self.device == 'cuda':
            inputs = inputs.to("cuda")

        out = self.model.generate(**inputs, max_length=200)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def classify_crime(self, caption):
        crime_keywords = ["crime", "weapon", "gun", "knife", "violence", "fight", "blood", "beating", "beaten"]
        if any(keyword in caption.lower() for keyword in crime_keywords):
            return "Crime detected"
        return "No crime"

    def save_model(self, save_dir="saved_blip"):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
