from Detections import Detections
import cv2
import numpy as np
import base64

class BufferItem:
    def __init__(self, stream_id, results, image):
        self.stream_id: str = stream_id
        self.results: dict[str, Detections] = results
        self.image: str = image
        
    def __dict__(self):
        return {
            'stream_id': self.stream_id,
            'results': {k:[m.__dict__() if isinstance(m,Detections) else m for m in v ] for k,v in self.results.items()},
            'image': self.image,
        }
    
    def to_annotated_image(self, return_base64=False):
        # Decode base64 image to numpy array
        img_data = base64.b64decode(self.image)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_map = {
            "object_frame": (0,255, 0),
            "emotion_frame": (255,0, 0),
            "activity_frame": (0,0, 255),
        } 

        # Draw detections
        for frame_type, detections in self.results.items():
            for det in detections:
                if isinstance(det,Detections):
                    det = det.__dict__()
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(image, (x1, y1), (x2, y2), color_map[frame_type], 1)
                text = f"{frame_type} {det['label']} {det['confidence']:.2f}"
                cv2.putText(image, text, (x1, y1+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if return_base64:
            # Re-encode to base64 if needed
            _, encoded_img = cv2.imencode('.jpg', image)
            return base64.b64encode(encoded_img).decode('utf-8')
        
        return image  # numpy array if not base64