from scripts.preprocessing.FrameProcessor import FrameProcessor
from scripts.preprocessing.PreprocessingManager import PreprocessingManager
from scripts.pipelining.TaskDispatcher import TaskDispatcher
from scripts.videostreamhandling.streamhandler import StreamHandler

from scripts.obj_det.YOLODetector import YOLODetector
from scripts.ai_agents.legacy.scene_understanding_llava import OllavaSceneUnderstanding
from scripts.ai_agents.legacy.Anomaly import AnomalyAnalyzer

import time

def main():
    obj_model = YOLODetector('yolov8n.pt')
    scene_model = OllavaSceneUnderstanding('llava:7b')
    models = {
        'object_frame': obj_model,
        'scene_frame': scene_model
    }

    Anomaly_model = AnomalyAnalyzer('llama3.1:8b')

    TD = TaskDispatcher(models,Anomaly_model)
    FP = FrameProcessor()

    PM = PreprocessingManager(TD, FP)
    PM.running = True
    SH = StreamHandler([['tests/assets/output_video.mp4',0.5],['tests/assets/video2.mp4',0.5]],PM)

    SH.start_streams()
    time.sleep(1)
    PM.process_frames()