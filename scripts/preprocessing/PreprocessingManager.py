from queue import Queue
from scripts.preprocessing.FrameProcessor import FrameProcessor


class TaskDispatcher:
    def __init__(self):
        pass

    def dispatch(self, frame: dict[str, any]):
        for source in frame.keys():
            print(f"Processed frame from source '{source}' dispatched for further processing.")
            #TODO: Implement sending to the appropraite sources


class PreprocessingManager:
    def __init__(self, task_dispatcher: TaskDispatcher, frame_processor: FrameProcessor):
        self.frame_queue = Queue()
        self.task_dispatcher = task_dispatcher
        self.frame_processor = frame_processor
        self.running = True  # Signal to control processing loop

    def add_frame(self, frame, source):
        if frame is None or source is None:
            raise ValueError("Frame and source cannot be None.")
        self.frame_queue.put((frame, source))

    def process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                try:
                    frame, source = self.frame_queue.get()
                    print(f"Processing frame from source '{source}'...")
                    processed_frame = self.frame_processor.process_frame(frame)
                    self.task_dispatcher.dispatch({source: processed_frame})
                except Exception as e:
                    print(f"Error processing frame: {e}")
    
    def stop_processing(self):
        self.running = False
        print("Processing loop stopped.")

        
    
