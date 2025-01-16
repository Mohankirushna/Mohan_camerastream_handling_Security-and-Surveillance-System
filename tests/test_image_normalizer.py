import numpy as np
import cv2
import sys
import os

# Ensure the scripts/preprocessing directory is in the Python path for importing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'preprocessing'))

import image_normalizer

def test_normalize_image_for_yolo():
    # Create a dummy image with random pixel values
    dummy_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    target_size = (416, 416)
    normalized = image_normalizer.normalize_image_for_yolo(dummy_image, target_size)
    
    # Check if the output shape matches target dimensions
    assert normalized.shape == (416, 416, 3), "Normalization failed: Incorrect shape."
    # Check pixel value range is between 0 and 1
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0, "Normalization failed: Pixel values out of range."

def test_denoise_and_sharpen():
    # Create a dummy noisy image
    dummy_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    output_image = image_normalizer.denoise_and_sharpen(dummy_image)
    
    # Check if the output shape remains the same as input
    assert output_image.shape == dummy_image.shape, "Denoise/Sharpen failed: Shape mismatch."

def test_preprocess_image_for_yolo():
    # Create another dummy image
    dummy_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    processed = image_normalizer.preprocess_image_for_yolo(dummy_image)
    
    # Validate shape and pixel range after full preprocessing
    assert processed.shape == (416, 416, 3), "Preprocessing failed: Incorrect shape."
    assert processed.min() >= 0.0 and processed.max() <= 1.0, "Preprocessing failed: Pixel values out of range."

if __name__ == "__main__":
    test_normalize_image_for_yolo()
    test_denoise_and_sharpen()
    test_preprocess_image_for_yolo()
    print("All tests passed.")
