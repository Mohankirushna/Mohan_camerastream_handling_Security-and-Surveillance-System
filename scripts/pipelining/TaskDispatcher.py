import cv2


class TaskDispatcher:
    def __init__(self, dispatch_models: dict[str,any], anomaly_model):
        self.dispatching_models = dispatch_models
        self.anomaly_models = anomaly_model

    def dispatch(self, frame: dict[str, dict]):
        for source,value in frame.items():
            print(f"Processed frame from source '{source}' dispatched for further processing.")

            result_obj = self.dispatching_models['object_frame'].detect( value['object_frame'] )
            result_scene = self.dispatching_models['scene_frame'].detect( value['scene_frame'] )
            
            print(result_obj)
            print(result_scene)

            output = self.anomaly_models.analyze(result_scene, result_obj)

            print(output)