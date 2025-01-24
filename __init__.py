from tests import test_image_normalizer, YOLODetectorTest, scene_understanding_test

scene_understanding_test.test_clip_model() # 605MB model
scene_understanding_test.test_blip_model() # 990MB model

print("-"*50, "Running test for YOLO image processing", "-"*50)
test_image_normalizer.test_normalize_image_for_yolo()
test_image_normalizer.test_denoise_and_sharpen()
test_image_normalizer.test_preprocess_image_for_yolo()
print("All tests passed.")

print("-"*50, "Running test for YOLO object detection", "-"*50)
YOLODetectorTest.YOLODetectorTest('tests/YOLODetector_test.jpg', ['yolov8n.pt'])