from scripts.obj_det.YOLODetector import YOLODetector

def YOLODetectorTest(image_path: any, model_paths:list[str])->dict:
    detectors = {model_path: YOLODetector(model_path) for model_path in model_paths}

    results_dict = {}
    for model_name, detector in detectors.items():
        print(f"Running inference with model: {model_name}")
        results = detector.detect(image_path)
        results_dict[model_name] = results

        # detector.visualize_results(image_path, results)

    print(results_dict)
