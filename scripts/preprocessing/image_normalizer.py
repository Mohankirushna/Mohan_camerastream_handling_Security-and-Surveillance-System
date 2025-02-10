import cv2
import numpy as np

def normalize_image_for_yolo(image, target_size=(416, 416)):
    """
    Resize and normalize the image for YOLO model input.
    
    Args:
        image (np.ndarray): Input image.
        target_size (tuple): Target resolution for YOLO, defaults to (416, 416).
    
    Returns:
        np.ndarray: Normalized image ready for YOLO.
    """
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    # Convert pixel values to float32 and scale them to [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    return normalized_image

def denoise_and_sharpen(image):
    """
    Remove noise from the image and apply sharpening to enhance features.
    
    Args:
        image (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Denoised and sharpened image.
    """
    # Apply a bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    # Apply the sharpening filter
    sharpened = cv2.filter2D(filtered, -1, kernel)
    return sharpened

def preprocess_image_for_yolo(image, target_size=(416, 416)):
    """
    Full preprocessing pipeline for YOLO: denoise, sharpen, resize and normalize.
    
    Args:
        image (np.ndarray): Input image.
        target_size (tuple): Target resolution for YOLO, defaults to (416, 416).
    
    Returns:
        np.ndarray: Preprocessed image ready for YOLO inference.
    """
    # Remove noise and sharpen the image
    cleaned_image = denoise_and_sharpen(image)
    # Normalize the cleaned image for YOLO
    normalized_image = normalize_image_for_yolo(cleaned_image, target_size)
    return normalized_image
