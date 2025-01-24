import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts','obj_det'))

from YOLODetector import YOLODetector

def YOLODetectorTest(image_path):
    model_paths = ['yolov8n.pt']
    detectors = {model_path: YOLODetector(model_path) for model_path in model_paths}

    results_dict = {}
    for model_name, detector in detectors.items():
        print(f"Running inference with model: {model_name}")
        results = detector.detect(image_path)
        results_dict[model_name] = results

        detector.visualize_results(image_path, results)

    print(results_dict)
