import cv2
import time
import asyncio
import numpy as np

class VideoStreamHandler:
    def __init__(self, video_sources, fps=2, max_retries=3, retry_interval=5):
        self.video_sources = video_sources
        self.fps = fps
        self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}
        self.frame_display_time = int(1000 / fps)
        self.video_order = list(video_sources.keys())
        self.frame_counters = {idx: 0 for idx in video_sources.keys()}
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def update_priority(self, camera_id, level):
        increase_factor = level + 1
        index_position = self.video_order.index(camera_id)
        self.video_order = [e for e in self.video_order if e != camera_id]
        for _ in range(increase_factor):
            self.video_order.insert(index_position + (len(self.video_order) // increase_factor), camera_id)
        print(f"Updated priority: {self.video_order}")

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
                return True
            print(f"[Camera {camera_id}] Reconnection attempt {attempt} failed.")

        print(f"[Camera {camera_id}] Could not reconnect after {self.max_retries} attempts.")
        return False

    async def handle_stream(self):
        index = 0
        while True:
            self.update_priority(3, 1)
            self.update_priority(5, 4)

            video_id = self.video_order[index]
            camera = self.cameras[video_id]

            if not camera.isOpened():
                print(f"[ERROR] Cannot open {self.video_sources[video_id]}. Trying to reconnect...")
                if not await self.reconnect_camera(video_id):
                    index = (index + 1) % len(self.video_order)
                    continue

            print(f"Now playing: Video {video_id} - {self.video_sources[video_id]}")
            
            for _ in range(self.fps):
                success, frame = camera.read()
                self.preprocessing_manager.add_frame(frame, video_sources[index])
                if not success:
                    print(f"[INFO] Video {video_id} has ended or failed to read.")
                    break
                
                if self.is_blank_or_dark(frame):
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[WARNING] Blank or dark frame detected in video {video_id} at {timestamp}")

                frame_number = self.frame_counters[video_id]
                self.frame_counters[video_id] += 1
                height, width, _ = frame.shape
                cv2.putText(frame, str(frame_number), (width - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Playing Video {video_id}", frame)

                if cv2.waitKey(1) == ord('q'):
                    self.stop_streams()
                    return
                
                await asyncio.sleep(0.001)
            
            index = (index + 1) % len(self.video_order)

    async def start_streams(self):
        await self.handle_stream()

    def stop_streams(self):
        for camera in self.cameras.values():
            camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Streams stopped.")

#example
if __name__ == "__main__":
    video_sources = {
        1: r"C:\\Users\\svars\\Downloads\\video1.mp4",
        2: r"C:\\Users\\svars\\Downloads\\video2.mp4",
        3: r"C:\\Users\\svars\\Downloads\\video3.mp4",
        4: r"C:\\Users\\svars\\Downloads\\video4.mp4",
        5: r"C:\\Users\\svars\\Downloads\\video5.mp4",
        6: r"C:\\Users\\svars\\Downloads\\video6.mp4"
    }

    handler = VideoStreamHandler(video_sources, fps=2)
    asyncio.run(handler.start_streams())
