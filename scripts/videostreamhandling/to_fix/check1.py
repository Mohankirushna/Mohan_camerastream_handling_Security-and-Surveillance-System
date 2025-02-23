import cv2
import os
import numpy as np
import threading

def process_video_one_frame_per_two_seconds(video_paths):
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Error: The file {path} does not exist.")
            return
    
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more video files.")
        return
    
    fps = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    total_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    
    frame_counts = [0] * len(video_paths)
    
    cv2.namedWindow("Processed Videos", cv2.WINDOW_NORMAL)

    while True:
        frames = []
        for i, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counts[i])
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to read frame from video {i + 1}.")
                continue
            
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.resize(processed_frame, (320, 240))
            frames.append(processed_frame)
            
            frame_counts[i] += int(fps[i] * 2)
            
            if frame_counts[i] >= total_frames[i]:
                frame_counts[i] = 0
        
        if len(frames) > 1:
            combined_frame = np.hstack(frames)
        elif len(frames) == 1:
            combined_frame = frames[0]
        else:
            break
        
        cv2.imshow("Processed Videos", combined_frame)
        print(f"Active threads: {threading.active_count()}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def main():
    video_files = [
        "/Users/mohankirushna.r/Downloads/COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4",
        "/Users/mohankirushna.r/Downloads/videoplayback.mp4",
        "/Users/mohankirushna.r/Downloads/Elden Ring - 5 hours of new Gameplay (PS5).mp4",
        "/Users/mohankirushna.r/Downloads/4K Scenic Byway 12 _ All American Road in Utah, USA - 5 Hour of Road Drive with Relaxing Music (1).mp4",
        "/Users/mohankirushna.r/Downloads/videoplayback (1).mp4"
    ]
    
    print("Starting video processing with all frames in a single window...")
    process_video_one_frame_per_two_seconds(video_files)

if __name__ == "__main__":
    main()
