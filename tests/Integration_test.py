from scripts.preprocessing.FrameProcessor import FrameProcessor
from scripts.preprocessing.PreprocessingManager import PreprocessingManager, TaskDispatcher
from scripts.videostreamhandling.streamhandler import StreamHandler

from scripts.obj_det.YOLODetector import YOLODetector
from scripts.ai_agents.scene_understanding_blip_salesforce import SceneUnderstanding

import time

def main():
    obj_model = YOLODetector('yolov8n.pt')
    scene_model = SceneUnderstanding('cpu')
    models = {
        'object_frame': obj_model,
        'scene_frame': scene_model
    }
    TD = TaskDispatcher(models)
    FP = FrameProcessor()

    PM = PreprocessingManager(TD, FP)
    PM.running = True
    SH = StreamHandler([[0,1],['tests/assets/output_video.mp4',1]],PM)

    SH.start_streams()
    time.sleep(1)
    PM.process_frames()