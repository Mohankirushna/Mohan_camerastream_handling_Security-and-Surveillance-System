import time
import threading
import cv2

class StreamHandler:
    def __init__(self, sources:list, preprocessing_manager):
        self.check_inputs(sources)
        self.sources = sources
        self.threads = []
        self.preprocessing_manager = preprocessing_manager

    def check_inputs(self, sources:list):
        assert len(sources) > 0
        for source in sources:
            assert len(source) == 2

    def start_streams(self):
        for source,fps in self.sources:
            print(source,fps)
            thread = threading.Thread(target=self.handle_stream, args=(source,fps,))
            thread.start()
            self.threads.append(thread)

    def handle_stream(self, source, fps):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error")
        t0 = time.time()
        while time.time()-t0 < 10: # cap.isOpened()
            ret, frame = cap.read()
            if not ret:
                break
            self.preprocessing_manager.add_frame(frame, source)
            print("Frame added", self.preprocessing_manager.running)
            # if not self.preprocessing_manager.running:
            #     break
            time.sleep(1/fps)
        cap.release()
        # self.preprocessing_manager.running = False