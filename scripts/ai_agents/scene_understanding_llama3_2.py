import ollama
from PIL import Image
import io

class OllavaSceneUnderstanding:
    def __init__(self):
        self.model = "llama3.2-vision:latest"  # Ensure the LLaVA model is downloaded

    def load_image(self, image_path):
        """Load an image from the file path."""
        return Image.open(image_path).convert("RGB")

    def analyze_scene(self, image_path):
        """Processes an image and asks Ollama to explicitly confirm if crime is detected."""
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

        # Generate response
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are an AI security analyst reviewing surveillance images for potential crime detection."},
            {"role": "user", "content": prompt, "images": [image_bytes]}
        ])

        # Extract text response
        caption = response['message']['content']
        return caption  # Ollama now decides if crime is present

# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\kvrus\OneDrive\Desktop\Test2.jpg"  # Use raw string (r"...") for Windows paths
    
    ollava_analyzer = OllavaSceneUnderstanding()
    caption = ollava_analyzer.analyze_scene(image_path)
    
    print("Ollama Response:\n", caption)