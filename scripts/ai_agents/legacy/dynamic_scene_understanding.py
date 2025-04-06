import os
import ollama
from PIL import Image
import io

class OllamaSceneUnderstanding:
    def __init__(self, model="llava:13b"):
        self.model = model  # Ensure LLaVA is installed in Ollama

    def load_images_from_folder(self, folder_path, max_images=5):
        """
        Loads up to `max_images` from the given folder and returns a list of PIL images.
        """
        images = []
        valid_extensions = (".jpg", ".jpeg", ".png")  # Supported formats

        # Get all image files and sort them (to maintain order)
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)])

        for image_file in image_files[:max_images]:  # Load only max_images
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path).convert("RGB")
            images.append(img)

        return images

    def analyze_scene(self, frames):
        """
        Sends multiple images to LLaVA for scene understanding.
        """
        image_bytes_list = []
        for image in frames:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            image_bytes_list.append(img_bytes.getvalue())

        # Scene understanding prompt
        prompt = (
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

        # Send request to LLaVA
        response = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": "You are analyzing images to understand dynamic scenes."},
            {"role": "user", "content": prompt, "images": image_bytes_list}
        ])

        return response['message']['content']

# Example usage
if __name__ == "__main__":
    folder_path = r"C:\Users\kvrus\OneDrive\Desktop\Projects\Python\GDSC\Models\VideoImages"  # Update with your folder path

    analyzer = OllamaSceneUnderstanding()
    frames = analyzer.load_images_from_folder(folder_path, max_images=5)

    if frames:
        result = analyzer.analyze_scene(frames)
        print("Scene Understanding Response:\n", result)
    else:
        print("No images found. Check your folder path.")
