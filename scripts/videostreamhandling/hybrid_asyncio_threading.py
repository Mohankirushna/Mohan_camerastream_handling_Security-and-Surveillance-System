import cv2
import time
import asyncio
import numpy as np
import threading

class VideoStreamHandler:
    def __init__(self, video_sources, fps=1, max_retries=3, retry_interval=5, debug_number=1):
        self.video_sources = video_sources
        self.debug_number = debug_number
        self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}
        self.videoflags = {idx: True for idx in video_sources.keys()}
        self.running = True
        self.fps = fps
        self.wrongcameras = set()
        self.wrongcounter = {idx: 3 for idx in video_sources.keys()}
        self.video_order = list(video_sources.keys())
        self.frame_counters = {idx: 0 for idx in video_sources.keys()}
        self.max_retries = max_retries
        self.retry_interval = retry_interval

        if self.debug_number == 1:
            self.update_priority(3, 1)
            self.update_priority(5, 3)
            self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}

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

    async def reconnect_camera(self):
        while self.running:
            if not self.wrongcameras:
                await asyncio.sleep(2)
                continue
            
            for camera_id in list(self.wrongcameras):
                print(f"[Camera {camera_id}] Attempting to reconnect...")
                self.cameras[camera_id] = cv2.VideoCapture(self.video_sources[camera_id])
                
                if self.cameras[camera_id].isOpened():
                    print(f"[Camera {camera_id}] Reconnected successfully!")
                    self.videoflags[camera_id] = True
                    self.wrongcameras.remove(camera_id)
                    self.update_priority(camera_id, 0)
                else:
                    self.wrongcounter[camera_id] -= 1
                    if self.wrongcounter[camera_id] <= 0:
                        print(f"[Camera {camera_id}] Could not reconnect after {self.max_retries} attempts.")
                        self.wrongcameras.remove(camera_id)
                    print(f"[Camera {camera_id}] Reconnection attempt failed.")
                
                await asyncio.sleep(self.retry_interval)

    def handle_stream(self, video_id):
        while self.running:
            if video_id not in self.video_order:
                time.sleep(0.001)
                continue

            camera = self.cameras[video_id]
            if not self.videoflags[video_id] or camera is None or not camera.isOpened():
                print(f"[ERROR] Cannot open {self.video_sources[video_id]}. Trying to reconnect...")
                self.videoflags[video_id] = False
                self.update_priority(video_id, -1)
                self.wrongcameras.add(video_id)
                self.wrongcounter[video_id] = 3
                time.sleep(0.001)
                continue

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
            cv2.putText(frame, str(frame_number), (width - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.debug_number != 0:
                cv2.imshow(f"Playing Video {video_id}", frame)
                if cv2.waitKey(1) == ord('q'):
                    self.running = False
                    self.stop_streams()
                    return
            
            time.sleep(0.001)

    def start_streams(self):
        threads = [threading.Thread(target=self.handle_stream, args=(vid,), daemon=True) for vid in self.video_sources.keys()]
        for thread in threads:
            thread.start()
        
        asyncio.run(self.reconnect_camera())

    def stop_streams(self):
        self.running = False
        for camera in self.cameras.values():
            camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Streams stopped.")

if __name__ == "__main__":
    video_sources = {
        1: r"C:\Zlearning2024\GDG\video1.mp4",
        2: r"C:\Zlearning2024\GDG\video2.mp4",
        3: r"C:\Zlearning2024\GDG\video3.mp4",
        4: r"C:\Zlearning2024\GDG\video4.mp4",
        5: r"C:\Zlearning2024\GDG\video5.mp4"
    }
    
    handler = VideoStreamHandler(video_sources, fps=1)
    handler.start_streams()
