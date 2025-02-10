import numpy as np

from scripts.preprocessing.image_normalizer import normalize_image_for_yolo, denoise_and_sharpen, preprocess_image_for_yolo

def test_normalize_image_for_yolo():
    # Create a dummy image with random pixel values
    dummy_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    target_size = (416, 416)
    normalized = normalize_image_for_yolo(dummy_image, target_size)
    
    # Check if the output shape matches target dimensions
    assert normalized.shape == (416, 416, 3), "Normalization failed: Incorrect shape."
    # Check pixel value range is between 0 and 1
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0, "Normalization failed: Pixel values out of range."

def test_denoise_and_sharpen():
    # Create a dummy noisy image
    dummy_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
    output_image = denoise_and_sharpen(dummy_image)
    
    # Check if the output shape remains the same as input
    assert output_image.shape == dummy_image.shape, "Denoise/Sharpen failed: Shape mismatch."

def test_preprocess_image_for_yolo():
    # Create another dummy image
    dummy_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    processed = preprocess_image_for_yolo(dummy_image)
    
    # Validate shape and pixel range after full preprocessing
    assert processed.shape == (416, 416, 3), "Preprocessing failed: Incorrect shape."
    assert processed.min() >= 0.0 and processed.max() <= 1.0, "Preprocessing failed: Pixel values out of range."

