import cv2
import numpy as np
from refactored_scene_understanding import *


system_prompt = (
    "You are a security analyst. Provide assessment in EXACTLY this format:\n"
    "Security Concern: <concern>\n"
    "Reason: <reason>\n"
    "Level: <low|medium|high>\n"
    "Confidence: <percentage>%\n\n"
    "Base analysis on these inputs:"
)

usr_prompt = (
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

scene_model: OllamaSceneUnderstanding = create_scene_understanding('llama3', prompt=system_prompt)


img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

payload = {
                "stream_id": "stream1",
                "object_frame": [
                    {"confidence":0.89, "label": "person", "bbox": [100, 0, 150, 150]}
                ], # list[Detections]
                "activity_frame": [{"confidence":0.69, "label": "walking", "bbox": [120, 0, 170, 100]}],
                "emotion_frame": [],
                "image": img
            }

_, obj = scene_model._process_features("", payload['object_frame'])
_, act = scene_model._process_features("", payload['activity_frame'])
_, emo = scene_model._process_features("", payload['emotion_frame'])

combined = "\n".join([obj,act,emo])

print(obj)
print(act)
print(emo)
print(combined)

result = scene_model.analyze_scene(img,usr_prompt)