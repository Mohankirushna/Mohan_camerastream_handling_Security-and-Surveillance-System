import cv2
import time

def process_video_one_frame_per_minute(video_path, video_index):
    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    # Get video properties (fps, total frames, etc.)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    
    # Process one frame per minute
    minute_count = 1
    while True:
        frame_position = int(fps * 60 * (minute_count - 1))  # Calculate frame position for each minute
        if frame_position >= total_frames:
            break
        
        # Set the position of the video to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        # Read the frame at that position
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame at minute {minute_count} from {video_path}")
            break
        
        # Process the frame (you can add your processing here)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Example: Convert to grayscale
        
        # Display the processed frame (optional, you can remove this line to speed up processing)
        cv2.imshow(f"Processed Frame at minute {minute_count} - Video {video_index}", processed_frame)
        
        # Delay for a short time (optional, remove this line if you don't need a delay)
        cv2.waitKey(1)  # To automatically move to the next frame without user input
        
        # Move to the next minute
        minute_count += 1

    cap.release()

def main():
    video_files = [
        "/Users/mohankirushna.r/Downloads/COSTA RICA IN 4K 60fps HDR (ULTRA HD).mp4",  # Replace with your video file paths
        "/Users/mohankirushna.r/Downloads/videoplayback.mp4",  # Another video file
        # Add more videos here if needed
    ]
    
    # Iterate through each video in the list
    for idx, video_file in enumerate(video_files, start=1):
        print(f"Processing Video {idx}: {video_file}...")
        process_video_one_frame_per_minute(video_file, idx)
        print(f"Finished processing Video {idx}\n")
    
    print("All videos processed successfully!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting automated video processing...")
    main()
