class Detections:
    def __init__(self, label, confidence, bbox):
        self.confidence: float = confidence
        self.label: str = label
        self.bbox: list[float] = bbox
    
    def __dict__(self):
        return {
            'confidence': self.confidence,
            'label': self.label,
            'bbox': self.bbox,
        }