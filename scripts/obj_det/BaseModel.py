class BaseModel:
    def __init__(self):
        self.model = None
        self.out = {}

    def detect(self, frame):
        output: dict[str,list[dict[str,any]]] = {'detections':[{}]}
        self.out = output
        return output