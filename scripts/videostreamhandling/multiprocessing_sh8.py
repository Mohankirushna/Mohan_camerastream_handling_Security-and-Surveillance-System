import cv2
import time
import asyncio
import numpy as np

class VideoStreamHandler:
    def __init__(self, video_sources, preprocessing_manager, fps=1, max_retries=3, retry_interval=5):
        self.video_sources = video_sources
        self.preprocessing_manager = preprocessing_manager
        self.fps = fps
        self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}
        self.videoflags = {idx: True for idx in video_sources.keys()}
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.SHOW = False

    def update_priority(self, camera_id, level):
        print(f"Priority update called for Camera {camera_id} with level {level}, but handled via async tasks.")

    def is_blank_or_dark(self, frame, threshold=10, std_dev_threshold=5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        return mean_brightness < threshold or std_dev < std_dev_threshold

    async def reconnect_camera(self, camera_id):
        print(f"[Camera {camera_id}] Attempting to reconnect...")
        for attempt in range(1, self.max_retries + 1):
            await asyncio.sleep(self.retry_interval)
            self.cameras[camera_id] = cv2.VideoCapture(self.video_sources[camera_id])
            if self.cameras[camera_id].isOpened():
                print(f"[Camera {camera_id}] Reconnected successfully!")
                self.videoflags[camera_id] = True
                return True
            print(f"[Camera {camera_id}] Reconnection attempt {attempt} failed.")
        print(f"[Camera {camera_id}] Could not reconnect after {self.max_retries} attempts.")
        return False

    async def process_camera(self, camera_id):
        while True:
            if not self.videoflags[camera_id]:
                await asyncio.sleep(1)
                continue

            camera = self.cameras[camera_id]
            if not camera.isOpened():
                print(f"[ERROR] Cannot open {self.video_sources[camera_id]}. Trying to reconnect...")
                self.videoflags[camera_id] = False
                await self.reconnect_camera(camera_id)
                continue

            success, frame = camera.read()
            if not success:
                print(f"[INFO] Video {camera_id} has ended or failed to read.")
                break

            if self.is_blank_or_dark(frame):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[WARNING] Blank or dark frame detected in video {camera_id} at {timestamp}")

            self.preprocessing_manager.add_frame(frame, self.video_sources[camera_id])

            if self.SHOW:
                cv2.imshow(f"Camera {camera_id}", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.stop_streams()
                    return

            await asyncio.sleep(1 / self.fps)

    async def start_streams(self):
        tasks = [self.process_camera(cam_id) for cam_id in self.video_sources.keys()]
        await asyncio.gather(*tasks)

    def stop_streams(self):
        for camera in self.cameras.values():
            camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Streams stopped.")

if __name__ == "__main__":
    video_sources = {
        1: "0",
        3: "0",
        2: "scripts/videostreamhandling/video4.mp4"
    }

    class TestProcessingManager:
        def add_frame(self, frame, source):
            pass

    PM = TestProcessingManager()
    handler = VideoStreamHandler(video_sources, PM, fps=1)
    handler.SHOW = True
    asyncio.run(handler.start_streams())
