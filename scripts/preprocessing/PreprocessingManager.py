from queue import Queue
from scripts.preprocessing.FrameProcessor import FrameProcessor
from scripts.pipelining.TaskDispatcher import TaskDispatcher

class PreprocessingManager:
    def __init__(self, task_dispatcher: TaskDispatcher, frame_processor: FrameProcessor):
        self.frame_queue = Queue()
        self.task_dispatcher = task_dispatcher
        self.frame_processor = frame_processor
        self.running = False  # Signal to control processing loop

    def add_frame(self, frame, source):
        if frame is None or source is None:
            raise ValueError("Frame or source cannot be None.")
        self.frame_queue.put((frame, source))
        print("frame_put")

    def process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                try:
                    frame, source = self.frame_queue.get()
                    print("Qsize:",self.frame_queue.qsize())
                    print(f"Processing frame from source '{source}'...")
                    processed_frame = self.frame_processor.process_frame(frame)
                    print("Frame processed")
                    self.task_dispatcher.dispatch({source: processed_frame})
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    self.stop_processing()
            else:
                # print("Empty")
                pass
    
    def stop_processing(self):
        self.running = False
        print("Processing loop stopped.")
