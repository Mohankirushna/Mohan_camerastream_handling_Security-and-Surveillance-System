import cv2
import time
import asyncio
import os
import numpy as np

class VideoStreamHandler:
    def __init__(self, video_sources, fps=1, max_retries=3, retry_interval=5, reconnect_duration=10):
        self.video_sources = video_sources
        self.fps = fps
        self.cameras = {idx: cv2.VideoCapture(path) if idx != 2 else None for idx, path in video_sources.items()}  # Video 2 starts disconnected
        self.video_order = list(video_sources.keys())
        self.frame_counters = {idx: 0 for idx in video_sources.keys()}
        self.videoflags = {idx: (idx != 2) for idx in video_sources.keys()}  
        self.retry_interval = retry_interval
        self.reconnect_duration = reconnect_duration

    def is_blank_or_dark(self, frame, threshold=10, std_dev_threshold=5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        return mean_brightness < threshold or std_dev < std_dev_threshold

    async def reconnect_camera(self, camera_id):
        """Reconnects video2.mp4 after 10 seconds."""
        if camera_id == 2 and self.cameras[camera_id] is None:
            print(f"[Camera {camera_id}] Simulated disconnection. Will reconnect after 10 seconds...")
            await asyncio.sleep(self.reconnect_duration)  

        print(f"[Camera {camera_id}] Attempting to reconnect...")
        start_time = time.time()

        while time.time() - start_time < self.reconnect_duration:
            await asyncio.sleep(self.retry_interval)

            if not os.path.exists(self.video_sources[camera_id]):
                print(f"[Camera {camera_id}] Video file still missing. Retrying...")
                continue

            self.cameras[camera_id] = cv2.VideoCapture(self.video_sources[camera_id])
            if self.cameras[camera_id].isOpened():
                print(f"[Camera {camera_id}] Video 2 successfully reconnected!")
                self.videoflags[camera_id] = True
                return True

            print(f"[Camera {camera_id}] Reconnection attempt failed. Retrying...")

        print(f"[Camera {camera_id}] Could not reconnect after {self.reconnect_duration} seconds.")
        return False

    async def handle_stream(self):
        index = 0
        asyncio.create_task(self.reconnect_camera(2))  
        while True:
            video_id = self.video_order[index]
            if not self.videoflags[video_id]:
                index = (index + 1) % len(self.video_order)
                continue

            camera = self.cameras[video_id]
            if camera is None or not camera.isOpened():
                print(f"[ERROR] Cannot open {self.video_sources[video_id]}. Trying to reconnect...")
                if not await self.reconnect_camera(video_id):
                    index = (index + 1) % len(self.video_order)
                    continue
                await asyncio.sleep(0.5)

            print(f"Now playing: Video {video_id} - {self.video_sources[video_id]}")
            success, frame = camera.read()
            if not success:
                print(f"[INFO] Video {video_id} has ended or failed to read.")
                break

            if self.is_blank_or_dark(frame):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[WARNING] Blank or dark frame detected in video {video_id} at {timestamp}")

            frame_number = self.frame_counters[video_id]
            self.frame_counters[video_id] += 1
            height, width, _ = frame.shape
            cv2.putText(frame, f"Frame {frame_number}", (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Playing Video {video_id}", frame)

            if cv2.waitKey(100) == ord('q'):
                self.stop_streams()
                return

            await asyncio.sleep(0.001)
            index = (index + 1) % len(self.video_order)

    async def start_streams(self):
        await self.handle_stream()

    def stop_streams(self):
        for camera in self.cameras.values():
            if camera:
                camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Streams stopped.")
        exit()

if __name__ == "__main__":
    video_sources = {
        1: "/Users/mohankirushna.r/Downloads/video1.mp4",
        2: "/Users/mohankirushna.r/Downloads/video2.mp4",
        3: "/Users/mohankirushna.r/Downloads/video3.mp4",
        4: "/Users/mohankirushna.r/Downloads/video4.mp4",
        5: "/Users/mohankirushna.r/Downloads/video5.mp4"
    }

    handler = VideoStreamHandler(video_sources, fps=2)
    asyncio.run(handler.start_streams())
