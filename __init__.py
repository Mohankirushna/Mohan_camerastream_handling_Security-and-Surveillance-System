from tests import test_image_normalizer, YOLODetectorTest

test_image_normalizer.test_normalize_image_for_yolo()
test_image_normalizer.test_denoise_and_sharpen()
test_image_normalizer.test_preprocess_image_for_yolo()
print("All tests passed.")

YOLODetectorTest.YOLODetectorTest('tests/YOLODetector_test.jpg')