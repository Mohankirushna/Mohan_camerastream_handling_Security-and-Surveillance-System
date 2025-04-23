import cv2
import numpy as np
from refactored_scene_understanding import *
from logger import logger

system_prompt = (
    "You are a security analyst."
)
usr_prompt = (
    "Provide assessment in EXACTLY this format:\n"
    "Security Concern: <concern>\n"
    "Reason: <reason>\n"
    "Level: <low|medium|high>\n"
    "Confidence: <percentage>%\n\n"
    "Base analysis on these inputs:"
)
scene_model: OllamaSceneUnderstanding = create_scene_understanding(model_type='llama3.1:8b', prompt=system_prompt)


# img = cv2.imread("test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

combined_det = "- person (Confidence: 89.0%) at [100, 0, 150, 150]\n- walking (Confidence: 69.0%) at [120, 0, 170, 100]"
scene_result = """This image depicts a man walking down a hallway captured by a security camera.

**People:**

*   One individual is visible, wearing a white collared shirt and dark pants.
*   He carries a black folder under his left arm.
*   The man walks towards the camera in a straight line, his head held low, with no discernible facial expression.

**Clothing:**

*   The individual wears a crisp white collared shirt.
*   His attire consists of dark pants that appear to be dress pants or trousers.

**Objects:**

*   There are no other objects visible in the image beyond the man and the hallway walls.

**Background:**

*   The background is a long, narrow hallway with plain white walls.
*   The floor appears to be made of wood or laminate.
*   The setting suggests an office environment, possibly a corporate building.

**Interactions:**

*   There are no visible interactions between people or objects in the scene.
*   The man's purpose for carrying the folder is unclear from this image alone.

**Summary:**

*   The scene begins with the man walking down the hallway towards the camera.
*   As the analysis progresses, it becomes clear that the man is likely an employee moving through the office building.
*   Without additional images or context, further details about his destination or purpose cannot be determined."""

payload = {
    "stream_id": "stream1",
    "detections": combined_det,
    "caption": scene_result
}


result = scene_model.analyze_scene(None, usr_prompt + f"Detections: {combined_det}\nSCENE: {scene_result}")

print(result)