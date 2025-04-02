import cv2
import time
import asyncio
import numpy as np
import threading
import os
import math
import csv
from itertools import zip_longest

class VideoStreamHandler:
    def __init__(self, video_sources, fps=1, max_retries=3, retry_interval=5, debug_number=1, thread_id=0):
        self.thread_id = thread_id
        self.video_sources = video_sources
        self.debug_number = debug_number

        # Make sure cameras are initialized correctly for all debug_number values
        if self.debug_number == 2:
            video_sources[2] = r""
            self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}
        elif self.debug_number == 0:
            self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}
        elif self.debug_number == 3:
            self.cameras = {idx: cv2.VideoCapture(path) if idx != 2 else None for idx, path in video_sources.items()}
        else:
            self.cameras = {idx: cv2.VideoCapture(path) for idx, path in video_sources.items()}  # Default case to ensure cameras are initialized

        self.videoflags = [True, True, True, True, True, True]
        self.running = True
        self.fps = fps
        self.wrongcameras = []
        self.wrongcounter = {}
        self.video_order = list(video_sources.keys())
        self.video_cloner = self.video_order.copy()
        self.frame_counters = {idx: 0 for idx in video_sources.keys()}
        self.max_retries = max_retries
        self.retry_interval = retry_interval


    def update_priority(self, camera_id, level):
        increase_factor = level + 1
        index_position = self.video_cloner.index(camera_id)
        self.video_cloner = [e for e in self.video_cloner if e != camera_id]
        for _ in range(increase_factor):
            self.video_cloner.insert(index_position + (len(self.video_cloner) // increase_factor), camera_id)
        print(f"Updated priority: {self.video_cloner}")

    def is_blank_or_dark(self, frame, threshold=10, std_dev_threshold=5):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        return mean_brightness < threshold or std_dev < std_dev_threshold

    async def reconnect_camera(self):
        global activecams
        while True:
            if not self.running or not activecams:
                break
            if len(self.wrongcameras) == 0:
                await asyncio.sleep(2)
                continue
            for attempt in range(1, self.max_retries + 1):
                print(self.wrongcameras)
                for camera_id in self.wrongcameras:
                    print(f"[Camera {camera_id}] Attempting to reconnect...")
                    self.cameras[camera_id] = cv2.VideoCapture(self.video_sources[camera_id])
                    if self.cameras[camera_id].isOpened():
                        print(f"[Camera {camera_id}] Reconnected successfully!")
                        self.videoflags[camera_id] = True
                        self.wrongcameras.remove(camera_id)
                        self.update_priority(camera_id, 0)
                        continue
                    if self.wrongcounter[camera_id] <= 0:
                        print(f"[Camera {camera_id}] Could not reconnect after {self.max_retries} attempts.")
                        self.wrongcameras.remove(camera_id)
                    else:
                        self.wrongcounter[camera_id] -= 1
                    print(f"[Camera {camera_id}] Reconnection attempt {attempt} failed.")

                await asyncio.sleep(self.retry_interval)

    async def apple(self):
        global activecams
        tasks = [self.handle_stream(cam_id) for cam_id in self.video_sources.keys()]
        tasks.append(self.reconnect_camera())
        await asyncio.gather(*tasks)
        print("task returns to apple")

    async def handle_stream(self, video_id):
        global activecams
        global lensources
        global max_threads
        global video_stamps

        # Check if cameras are initialized
        if not hasattr(self, 'cameras') or self.cameras is None:
            print("[ERROR] Cameras not initialized properly.")
            return

        while True:
            if not self.running or not activecams:
                break
            if len(self.video_order) == 0:
                self.video_order = self.video_cloner.copy()
                await asyncio.sleep(0.001)
                continue
            if video_id not in self.video_order:
                await asyncio.sleep(0.001)
                continue
            self.video_order.remove(video_id)
            trunum = self.thread_id + (video_id - 1) * max_threads
            camera = self.cameras.get(video_id, None)  # Use .get() to avoid direct access errors
            if not self.videoflags[video_id]:
                await asyncio.sleep(0.001)
                continue
            if camera is None or not camera.isOpened():
                print(f"[ERROR] Cannot open {self.video_sources[video_id]}. Trying to reconnect...")
                if self.videoflags[video_id]:
                    self.videoflags[video_id] = False
                    self.update_priority(video_id, -1)
                    self.wrongcameras.append(video_id)
                    self.wrongcounter[video_id] = 3
                await asyncio.sleep(0.001)
                continue

            success, frame = camera.read()
            if not success:
                print(f"[INFO] Video {video_id} has ended or failed to read.")
                break
            millis = int((time.time() % 1) * 1000)
            timestamp = time.strftime("%H:%M:%S") + ".{:03d}".format(millis)
            frame_number = self.frame_counters[video_id]
            self.frame_counters[video_id] += 1

            # Write to CSV log
            video_stamps[trunum-1].append(timestamp)

            # Display video frames
            height, width, _ = frame.shape
            cv2.putText(frame, str(frame_number), (width - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.debug_number != 0:
                cv2.imshow(f"Playing Video {trunum} from thread {self.thread_id}", frame)

                if cv2.waitKey(1) == ord('q'):
                    print("Exiting...")
                    activecams = False
                    self.running = False
                    self.stop_streams()

            await asyncio.sleep(0.001)

    async def start_streams(self):
        await self.handle_stream()

    def stop_streams(self):
        for camera in self.cameras.values():
            camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Streams stopped.")
        # Close the CSV file after the streams are stopped


    def threader(self):
        print(self.video_cloner)
        print(self.video_order)
        asyncio.run(self.apple())

if __name__ == "__main__":
    logical_cores = os.cpu_count()
    print(math.floor(0.75 * logical_cores))
    max_threads = math.floor(0.75 * logical_cores)
    max_threads = 8
    activecams = True

    video_sources = {
    1: r"C:\Zlearning2024\GDG\video1.mp4",
    2: r"C:\Zlearning2024\GDG\video2.mp4",
    3: r"C:\Zlearning2024\GDG\video3.mp4",
    4: r"C:\Zlearning2024\GDG\video4.mp4",
    5: r"C:\Zlearning2024\GDG\video5.mp4",
    6: r"C:\Zlearning2024\GDG\video1.mp4",
    7: r"C:\Zlearning2024\GDG\video2.mp4",
    8: r"C:\Zlearning2024\GDG\video3.mp4",
    9: r"C:\Zlearning2024\GDG\video4.mp4",
    10: r"C:\Zlearning2024\GDG\video5.mp4",
    11: r"C:\Zlearning2024\GDG\video1.mp4",
    12: r"C:\Zlearning2024\GDG\video2.mp4",
    13: r"C:\Zlearning2024\GDG\video3.mp4",
    14: r"C:\Zlearning2024\GDG\video4.mp4",
    15: r"C:\Zlearning2024\GDG\video5.mp4",
    16: r"C:\Zlearning2024\GDG\video1.mp4",
    17: r"C:\Zlearning2024\GDG\video2.mp4",
    18: r"C:\Zlearning2024\GDG\video3.mp4",
    19: r"C:\Zlearning2024\GDG\video4.mp4",
    20: r"C:\Zlearning2024\GDG\video5.mp4",
}
    video_headers = ["video "+str(i) for i in range(len(video_sources))]
    video_stamps = [list() for i in video_sources]

    lensources = len(video_sources)
    video_subsources = []
    loopvar = 0
    list_video_sources = list(video_sources.values())
    while len(list_video_sources) > 0:
        if loopvar == len(video_subsources):
            video_subsources.append(list())
        video_subsources[loopvar].append(list_video_sources.pop(0))
        loopvar = (loopvar + 1) % max_threads
        print(video_subsources)

    threads_holder = list()
    for i in video_subsources:
        converted_dict = dict()
        for j in range(len(i)):
            converted_dict[j + 1] = i[j]
        print(converted_dict)
        handler = VideoStreamHandler(converted_dict, fps=1, thread_id=len(threads_holder) + 1)
        threads_holder.append(threading.Thread(target=handler.threader))
        threads_holder[len(threads_holder) - 1].start()
        print("thread ", len(threads_holder) - 1, " has been started")
    for i in range(len(threads_holder)):
        threads_holder[i].join()
        print("Thread ", i, " has been joined")
    
        # Initialize the CSV file to log the frames processed
        csv_file = open(f'GDGSHData.csv', mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(video_headers)
        colms = list(zip_longest(*video_stamps, fillvalue=None))
        csv_writer.writerows(colms)
        csv_file.close()
    print(list(len(i) for i in video_stamps))
    print("whoawwww")
