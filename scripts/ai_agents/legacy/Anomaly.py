"""
ollama pull llama3.2-vision:11b   -- 7.9 GB
ollama pull llava:7b   -- 4.7 GB
ollama pull llava:34b   -- 20 GB
ollama pull llama3.1:8b   -- 4.9 GB
ollama pull llama2-uncensored:7b   -- 3.8 GB
ollama pull gemma2:27b   -- 16 GB
ollama pull openthinker:32b   -- 20 GB
ollama pull deepseek-v2:16b   -- 8.9 GB
ollama pull deepseek-r1:14b   -- 9 GB
ollama pull deepseek-r1:32b   -- 20 GB

or

echo llama3.2-vision:11b llava:7b llava:34b llama3.1:8b llama2-uncensored:7b gemma2:27b openthinker:32b deepseek-v2:16b deepseek-r1:14b deepseek-r1:32b | xargs -n1 ollama pull

# models to test
text_models = ["gemma2:27b", "llama2-uncensored:7b", "llama3.1:8b", "openthinker:32b", "deepseek-v2:16b", "deepseek-r1:14b", "deepseek-r1:32b"]
vision_models = ['llava:7b', 'llava:34b', 'llama3.2-vision:11b']
# inputs
input_features = ['scene_details', 'objects']
# example input
input = {
    'scene_details': 'ashdajskgd', # text str
    'objects': [ # detections list[dict[str,any]]
        {
            'class': 'person',
            'confidence' : 0.89,
            'bbox' : [0.1,0.1,0.4,0.2]
        },
    ]
}


    The model will look at what the scene understanding models said as well as the predictions of the object detection models.
    The model might also consider the 'frame' (this can be a segmented mask or just an annotated frame) via passing it to the vision model.
    All the details are then joined into one prompt and a vertdict is given
"""
import ollama

class AnomalyAnalyzer:
    def __init__(self, model):
        self.model = model

    def _generate_prompt_template(self, scene:str, objects:str):
        # system_prompt = (
        #     "You provide a detailed Security and Surveillance assessment with confidence level clearly showcasing: "
        #     "1) Security concern (few words or a sentence) 2) Reason for Security concern (100 words) 3) Level of Security concern (low,medium,high)"
        #     "Answer only in this template."
        # )
        system_prompt = (
            "You are a security analyst. Provide assessment in EXACTLY this format:\n"
            "Security Concern: <concern>\n"
            "Reason: <reason>\n"
            "Level: <low|medium|high>\n"
            "Confidence: <percentage>%\n\n"
            "Base analysis on these inputs:"
        )
        user_prompt = (
            f"Scene Context: {scene}\n"
            "Detected Objects:\n" + objects
        )
        return system_prompt, user_prompt

    def _process_features(self, scene: str, objects: list[dict])->tuple[str,str]:
        object_description = "\n".join(
            f"- {obj.get('label', 'unknown')} "
            f"(Confidence: {obj.get('confidence', 0)*100:.1f}%) "
            f"at {obj.get('bbox', [])}"
            for obj in objects['detections']
        )
        return scene,object_description
    
    def analyze(self,scene: str, objects: list[dict]):
        scene_processed, objects_processed = self._process_features(scene,objects)
        system,prompt = self._generate_prompt_template(scene_processed, objects_processed)
        result = ollama.generate(
            model = self.model,
            prompt = prompt,
            system = system,
            options = {
                "num_ctx": 2048,  # Reduce context window
                "num_gpu_layers": 20,    # Allocate more layers to GPU
                "temperature": 0  # Disable randomness
            },
        )
        return result['response']


    