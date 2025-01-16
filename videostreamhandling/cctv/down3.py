import cv2
import time
import numpy as np
import threading

def display_camera_details(camera_indices, priority_camera_index, priority_fps, other_fps, actual_fps, lock):
    """Thread function to display camera details."""
    while True:
        with lock:
            print("\n[Camera Details]")
            for i in range(len(camera_indices)):
                target_fps = priority_fps if i == priority_camera_index else other_fps
                print(f"Camera {i}: {'Priority Camera' if i == priority_camera_index else 'Normal Camera'} | "
                      f"Target FPS: {target_fps} | Actual FPS: {actual_fps[i]:.2f}")
        time.sleep(1)  # Update details every second

def calculate_fps(frame_counts, actual_fps, lock):
    """Thread function to calculate and update FPS every second."""
    while True:
        time.sleep(1)
        with lock:
            for i in range(len(frame_counts)):
                actual_fps[i] = frame_counts[i]
                frame_counts[i] = 0  # Reset frame count for the next interval

def process_webcam_streams(camera_indices, priority_camera_index=0, priority_fps=30, other_fps=1):
    """Process and display webcam streams with FPS prioritization."""
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more cameras.")
        return

    cv2.namedWindow("Processed Webcams", cv2.WINDOW_NORMAL)
    num_cameras = len(caps)

    # Time intervals for frame capture
    intervals = [1 / (priority_fps if i == priority_camera_index else other_fps) for i in range(num_cameras)]
    last_frame_times = [0] * num_cameras  # Tracks the last frame capture times
    frame_counts = [0] * num_cameras      # Tracks frame counts for FPS calculation
    actual_fps = [0.0] * num_cameras      # Stores actual FPS for each camera

    lock = threading.Lock()  # Lock for thread-safe access to shared data

    # Start FPS calculation thread
    fps_thread = threading.Thread(target=calculate_fps, args=(frame_counts, actual_fps, lock))
    fps_thread.daemon = True
    fps_thread.start()

    while True:
        frames = []
        current_time = time.time()

        for i in range(num_cameras):
            interval = intervals[i]

            if current_time - last_frame_times[i] >= interval:
                last_frame_times[i] = current_time  # Update the last frame time

                ret, frame = caps[i].read()
                if not ret:
                    print(f"Camera {i}: Failed to read frame.")
                    processed_frame = np.zeros((240, 320), dtype=np.uint8)  # Blank frame for failed reads
                else:
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    processed_frame = cv2.resize(processed_frame, (320, 240))  # Resize frame
                    frame_counts[i] += 1  # Increment frame count

                # Add a label to the frame
                cv2.putText(processed_frame, f"Camera {i}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                frames.append(processed_frame)
            else:
                # Append a blank frame for skipped intervals
                frames.append(np.zeros((240, 320), dtype=np.uint8))

        # Combine frames for display
        combined_frame = np.hstack(frames)
        cv2.imshow("Processed Webcams", combined_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture objects
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to set up and run the webcam processing."""
    camera_indices = [0, 1]  # Indices for the two cameras
    priority_camera_index = 0  # Camera 0 is the priority camera
    priority_fps = 30  # Target FPS for the priority camera
    other_fps = 1      # Target FPS for normal cameras

    print("Starting webcam stream processing with dynamic FPS handling...")

    # Shared list for actual FPS values
    actual_fps = [0.0] * len(camera_indices)

    # Lock for thread-safe data access
    lock = threading.Lock()

    # Start the camera details thread
    details_thread = threading.Thread(
        target=display_camera_details,
        args=(camera_indices, priority_camera_index, priority_fps, other_fps, actual_fps, lock)
    )
    details_thread.daemon = True
    details_thread.start()

    # Start processing webcam streams
    process_webcam_streams(camera_indices, priority_camera_index=priority_camera_index,
                           priority_fps=priority_fps, other_fps=other_fps)

if __name__ == "__main__":
    main()
