import os
import cv2
import time

def process_video_one_frame_per_two_seconds(video_paths):
    # Check if files exist before processing
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Error: The file {path} does not exist.")
            return
    
    # Open video files with OpenCV
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more video files.")
        return
    
    # Get video properties (fps, total frames, etc.)
    fps = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]  # Frames per second for each video
    total_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]  # Total number of frames
    
    # Store processed frames
    frame_counts = [0] * len(video_paths)  # Track the current frame position for each video
    
    # Create windows for displaying both videos
    for i in range(len(video_paths)):
        cv2.namedWindow(f"Processed Video {i+1}", cv2.WINDOW_NORMAL)

    # Process frames
    while True:
        for i, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counts[i])  # Set the current frame position
            
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to read frame from video {i + 1}.")
                continue
            
            # Example processing (grayscale conversion)
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Show processed frames (alternately in separate windows)
            cv2.imshow(f"Processed Video {i+1}", processed_frame)
            
            # Update frame position for 1 frame every 2 seconds
            frame_counts[i] += int(fps[i] * 2)  # Skip 2 seconds worth of frames
            
            if frame_counts[i] >= total_frames[i]:
                # Reset frame count to start over if we reach the end of the video
                frame_counts[i] = 0

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video captures and close windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def main():
    # Specify the video file paths
    video_files = [
        "/Users/mohankirushna.r/Downloads/COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4",  # Replace with your local video file path
        "/Users/mohankirushna.r/Downloads/videoplayback.mp4",  # Replace with another local video file path
        "/Users/mohankirushna.r/Downloads/Elden Ring - 5 hours of new Gameplay (PS5).mp4.crdownload",  # Make sure the file is fully downloaded
        "/Users/mohankirushna.r/Downloads/4K Scenic Byway 12 _ All American Road in Utah, USA - 5 Hour of Road Drive with Relaxing Music (1).mp4"
    ]
    
    print("Starting video processing with alternate frame processing...")
    process_video_one_frame_per_two_seconds(video_files)

if __name__ == "__main__":
    main()
