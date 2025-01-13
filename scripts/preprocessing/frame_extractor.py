import cv2
import numpy as np

# might possibly be needed based on how the cameras stream is set up
from queue import Queue
from threading import Thread 

class PreProcessor():
    def __init__(self):
        # TODO: implement better processing functions like histogram equalization
        # TODO: implment processing functions based on embeddings (CLIP models)
        pass
    def histogram_equilization(self,frame):
        """
        Function to improve the contrase of the frame: Allows lower constrase areas to gain a highet contrase.

        input argument: frame
        returns: Contrast enhanced frame
        """
        
    
    def preprocess_frame_for_scene_understanding(self, frame): 
        # basic frame processing for CLIP models
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        return frame_normalized

    def preprocess_frame_for_object_detection(self, frame):
        # basic frame processing for YOLO models
        frame_resized = cv2.resize(frame, (416, 416))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        return frame_normalized

    def process_frame(self, frame):
        scene_frame = self.preprocess_frame_for_scene_understanding(frame)
        object_frame = self.preprocess_frame_for_object_detection(frame)

        return {
            "scene_frame": scene_frame,
            "object_frame": object_frame
        }
    if __name__ = '__main__':
        preprocessor = PreProcessor()
        
    
