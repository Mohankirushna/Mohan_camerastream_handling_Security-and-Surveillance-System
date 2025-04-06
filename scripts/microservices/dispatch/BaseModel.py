from Detections import Detections

class BaseModel:
    def __init__(self):
        self.model = None
        self.out = {}

    def detect(self, frame):
        output: dict[str,list[Detections]] = {}
        self.out = output
        return output