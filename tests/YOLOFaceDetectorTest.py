import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'obj_det'))

from YOLOFaceDetector import YOLOFaceDetector  

def YOLOFaceDetectorTest(image_path):
    model_paths = [os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8_face_detection.pt')]
    detectors = {os.path.basename(model_path): YOLOFaceDetector(model_path) for model_path in model_paths}

    results_dict = {}
    for model_name, detector in detectors.items():
        print(f"Running inference with model: {model_name}")
        results = detector.detect_faces(image_path)
        results_dict[model_name] = results

        detector.visualize_results(image_path, results)

    print("Detection Results:", results_dict)