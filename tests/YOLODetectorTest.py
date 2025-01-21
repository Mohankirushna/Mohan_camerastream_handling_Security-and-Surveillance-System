from YOLODetector import YOLODetector

if __name__ == "__main__":
    model_paths = ['yolov8n.pt']
    detectors = {model_path: YOLODetector(model_path) for model_path in model_paths}

    image_path = '/content/lori-abby.jpg'

    results_dict = {}
    for model_name, detector in detectors.items():
        print(f"Running inference with model: {model_name}")
        results = detector.detect(image_path)
        results_dict[model_name] = results

        detector.visualize_results(image_path, results)

    print(results_dict)
