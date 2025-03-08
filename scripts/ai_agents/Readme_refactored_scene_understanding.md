# Refactored Scene Understanding Module

## Overview

This refactored module combines four different scene understanding approaches into a single coherent class hierarchy:

1. **CLIP-based scene understanding** - Uses OpenAI's CLIP to rank predefined captions based on image content
2. **BLIP-based scene understanding** - Uses Salesforce's BLIP to generate free-form captions for images
3. **LLaVA-based scene understanding** - Uses Ollama with LLaVA model for scene analysis
4. **Llama 3.2-based scene understanding** - Uses Ollama with Llama 3.2 Vision for more structured scene analysis


## Class hiearchy

BaseSceneUnderstanding
├── CLIPSceneUnderstanding
├── BLIPSceneUnderstanding
├── LLaVASceneUnderstanding
└── Llama3VisionSceneUnderstanding