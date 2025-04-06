from PIL import Image
import requests

from scripts.ai_agents.legacy.scene_understanding_blip_salesforce import SceneUnderstanding
from scripts.ai_agents.legacy.sceneunderstanding_clip_openai import SceneUnderstandingWithCLIP
from scripts.ai_agents.legacy.scene_understanding_llava import OllavaSceneUnderstanding

img_url = "https://www.mixedmartialarts.com/.image/c_limit%2Ccs_srgb%2Cq_auto:good%2Cw_700/MTg3NzUxNDMzMDM0OTM0MjQx/us-army-soldier-vs-israeli-soldier-in-knife-fight-contest.webp"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

def test_clip_model():
    print("-"*50, "Testing CLIP Model for ranking best captions", "-"*50)
    scene_understanding = SceneUnderstandingWithCLIP()
    text_prompts = [
        "a photograph of a peaceful park",
        "a crime scene with a weapon",
        "a family having a picnic",
        "a busy street with cars",
        "a violent fight between people",
    ]
    scene_understanding.set_image(raw_image)
    best_caption, confidence = scene_understanding.rank_captions(text_prompts)
    print(f"Caption: {best_caption} (Confidence: {confidence:.2f})")
    print(scene_understanding.classify_crime(best_caption))

def test_blip_model():
    print("-"*50, "Testing BLIP Model for generating best captions", "-"*50)
    scene_understanding = SceneUnderstanding()
    scene_understanding.set_image(raw_image)

    caption = scene_understanding.generate_caption()
    print(caption)
    print(scene_understanding.classify_crime(caption))

def test_ollava():
    ollava_analyzer = OllavaSceneUnderstanding()
    caption = ollava_analyzer.analyze_scene(raw_image)
    print("Ollava Response:\n", caption)
