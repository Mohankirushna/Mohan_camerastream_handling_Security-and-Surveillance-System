from Detections import Detections

class BufferItem:
    def __init__(self, stream_id, results, image):
        self.stream_id: str = stream_id
        self.results: dict[str, Detections] = results
        self.image: str = image # base64 encoded
        
    def __dict__(self):
        return {
            'stream_id': self.stream_id,
            'results': {k:[m.__dict__() for m in v ] for k,v in self.results.items()},
            'image': self.image,
        }