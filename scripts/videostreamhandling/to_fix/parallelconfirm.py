import cv2
import threading
from queue import Queue
import numpy as np

def process_frame(frame):
    # Example processing (grayscale conversion)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def stream_video_from_file(video_path, frame_queue):
    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    # Set video capture resolution (optional, lower resolution for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height to 480
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate to 30 FPS
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame from {video_path}")
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Add the processed frame to the queue
        frame_queue.put(processed_frame)
    
    cap.release()

def display_frames_opencv(frame_queues, window_names):
    # Create a window to display the video for each source
    for window_name in window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        for idx, frame_queue in enumerate(frame_queues):
            if not frame_queue.empty():
                processed_frame = frame_queue.get()
                if processed_frame is not None:
                    # Display the processed frame
                    cv2.imshow(window_names[idx], processed_frame)
                
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    # Specify the local video file paths
    video_files = [
        "/Users/mohankirushna.r/Downloads/COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4",  # Replace with your local video file path
        "/Users/mohankirushna.r/Downloads/videoplayback.mp4",  # Replace with another local video file path
    ]
    
    # Window names for displaying multiple videos
    window_names = [f"Processed Video {i + 1}" for i in range(len(video_files))]
    
    # Create a queue to hold processed frames for each video source
    frame_queues = [Queue() for _ in video_files]
    
    # Start processing each video file in a separate thread
    threads = []
    for idx, video_path in enumerate(video_files):
        stream_thread = threading.Thread(target=stream_video_from_file, args=(video_path, frame_queues[idx]))
        stream_thread.start()
        threads.append(stream_thread)
    
    # Display frames using OpenCV in the main thread
    display_frames_opencv(frame_queues, window_names)
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    print("Starting video processing...")
    main()
