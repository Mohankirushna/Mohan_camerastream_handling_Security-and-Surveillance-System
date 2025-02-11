import cv2
import numpy as np
import time
import threading

def process_webcam_streams(camera_indices, total_cameras=12, priority_camera_index=None, priority_interval=10):
    caps = [cv2.VideoCapture(index) for index in camera_indices]
    
    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more cameras.")
        return
    
    frame_counts = [0] * len(camera_indices)
    
    cv2.namedWindow("Processed Webcams", cv2.WINDOW_NORMAL)

    last_priority_time = time.time()  # To track time for priority frame processing
    
    # Define layout for 12 cameras (3 rows of 4 columns)
    rows = 3
    cols = 4
    window_width = 1280
    window_height = 960

    while True:
        frames = []

        for i in range(total_cameras):
            if i < len(caps):  # Process available cameras
                ret, frame = caps[i].read()
                if not ret:
                    print(f"Error: Failed to read frame from camera {i}.")
                    continue
                
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.resize(processed_frame, (320, 240))
                
                # If it's the priority camera, process it first
                if priority_camera_index is not None and i == priority_camera_index:
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    processed_frame = cv2.resize(processed_frame, (320, 240))
                    # Add "Priority Camera" label on top of the frame
                    cv2.putText(processed_frame, "Priority Camera", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                frames.append(processed_frame)
            else:
                # If there are fewer cameras, create an empty frame with a message
                empty_frame = np.zeros((240, 320), dtype=np.uint8)  # Black frame
                cv2.putText(empty_frame, "Camera Not Available", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                frames.append(empty_frame)

        # Create a grid for the frames (3 rows, 4 columns)
        grid_frames = []
        for r in range(rows):
            row_frames = frames[r*cols:(r+1)*cols]  # Get 4 frames for each row
            row_combined = np.hstack(row_frames)  # Horizontally stack frames for the row
            grid_frames.append(row_combined)

        # Stack all rows vertically to create the final window
        combined_frame = np.vstack(grid_frames)
        
        # Resize the final frame to fit the window
        combined_frame_resized = cv2.resize(combined_frame, (window_width, window_height))

        # Display the combined frame
        cv2.imshow("Processed Webcams", combined_frame_resized)
        
        # Print active thread count
        print(f"Active threads: {threading.active_count()}")
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def main():
    camera_indices = [0, 1]  # Indices for the two web cameras
    priority_camera_index = 0  # Set Camera 1 as the priority camera (index 0)
    priority_interval = 10  # Check the priority camera every 10 seconds
    print("Starting webcam stream processing with Camera 1 as the priority...")
    
    # Start processing with the webcam streams
    process_webcam_streams(camera_indices, total_cameras=12, priority_camera_index=priority_camera_index, priority_interval=priority_interval)

if __name__ == "__main__":
    main()
