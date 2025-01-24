from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class SceneUnderstandingWithCLIP:
    def __init__(self, ):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        try:
            self.model.to("cuda")
        except:
            print("Failed to load to GPU")
            pass

    def set_image(self, img: Image):
        self.raw_image = img

    def rank_captions(self, potential_captions):
        """
        Generates a similarity score for each potential_captions and returns the best matching caption.
        :param potential_captions: List of textual descriptions to match with the image.
        :return: Best matching text description.
        """
        inputs = self.processor(
            text=potential_captions, images=self.raw_image, return_tensors="pt", padding=True
        )
        if self.model.device.type == "cuda":
            inputs = inputs.to('cuda')
        # Compute similarity scores
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-to-text scores
        probs = logits_per_image.softmax(dim=1)  # Normalize scores to probabilities

        # Find the best match
        best_idx = probs.argmax(dim=1).item()
        return potential_captions[best_idx], probs[0, best_idx].item()
    
    def classify_crime(self, caption):
        crime_keywords = ["crime", "weapon", "gun", "knife", "violence", "fight", "blood"]
        if any(keyword in caption.lower() for keyword in crime_keywords):
            return "Crime detected"
        return "No crime"

    def save_model(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)