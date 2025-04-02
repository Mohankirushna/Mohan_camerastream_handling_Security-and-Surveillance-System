import cv2
import numpy as np

class FrameProcessor():
    def __init__(self):
        pass

    def preprocess_frame_for_scene_understanding(self, model, frame):
        if model == 'blip':
            processed_frame = cv2.resize(frame, (336,336))
            processed_frame = self.denoise_and_sharpen(processed_frame)
            # processed_frame = self.histogram_equilization(processed_frame)
        else:
            print("Mentioned Model not implemented yet. Proceeding with basic denoise and sharpen")
            processed_frame = self.denoise_and_sharpen(frame)
        return processed_frame

    def preprocess_frame_for_object_detection(self, model, frame):
        if model == 'yolo':
            processed_frame = self.preprocess_image_for_yolo(frame,(416,416))
        elif model == 'sam':
            print("SAM model not implemented yet. Proceeding with yolo preprocessing")
            processed_frame = self.preprocess_image_for_yolo(frame,(416,416))
        else:
            print("Mentioned Model not implemented yet. Proceeding with yolo preprocessing")
            processed_frame = self.preprocess_image_for_yolo(frame,(416,416))
    
        return processed_frame

    def process_frame(self, frame, scene_model="blip", object_model="yolo"):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scene_frame = self.preprocess_frame_for_scene_understanding(scene_model, frame)
        object_frame = self.preprocess_frame_for_object_detection(object_model, frame)

        return {
            "scene_frame": scene_frame,
            "object_frame": object_frame
        }

    def normalize_image_for_yolo(self, image: np.ndarray, target_size=(416, 416)):
        resized_image = cv2.resize(image, target_size)
        # normalized_image = resized_image.astype(np.float32) / 255.0
        return resized_image

    def denoise_and_sharpen(self, image):
        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        return sharpened

    def preprocess_image_for_yolo(self, image, target_size=(416, 416)):
        cleaned_image = self.denoise_and_sharpen(image)
        normalized_image = self.normalize_image_for_yolo(cleaned_image, target_size)
        return normalized_image

    def histogram_equilization(self, frame):
        if len(frame.shape) == 2:  # Grayscale image
            return cv2.equalizeHist(frame)
        else:  # Color image
            # Convert to YUV color space
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
            # Convert back to BGR color space
            return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)