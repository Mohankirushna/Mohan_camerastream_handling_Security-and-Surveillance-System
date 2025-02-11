import cv2
import time
import numpy as np
import threading


def display_camera_details(camera_indices, priority_camera_index, priority_fps, other_fps):
    while True:
        print("\n[Camera Details]")
        for i in range(len(camera_indices)):
            if i == priority_camera_index:
                print(f"Camera {i}: Priority Camera | FPS: {priority_fps}")
            else:
                print(f"Camera {i}: Normal Camera | FPS: {other_fps}")
        time.sleep(1)  # Update details every second


def process_webcam_streams(camera_indices, priority_camera_index=0, priority_fps=30, other_fps=1):
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more cameras.")
        return

    cv2.namedWindow("Processed Webcams", cv2.WINDOW_NORMAL)
    num_cameras = len(caps)
    current_camera_index = 0  # Start with the first camera

    priority_interval = 1 / priority_fps  # Time interval for the priority camera
    other_interval = 1 / other_fps        # Time interval for other cameras

    last_frame_times = [0] * num_cameras  # Track the last frame capture times

    while True:
        frames = []
        current_time = time.time()

        for i in range(num_cameras):
            interval = priority_interval if i == priority_camera_index else other_interval

            if current_time - last_frame_times[i] >= interval:
                last_frame_times[i] = current_time  # Update the last frame time

                ret, frame = caps[i].read()
                if not ret:
                    print(f"Error: Failed to read frame from camera {i}.")
                    processed_frame = np.zeros((240, 320), dtype=np.uint8)  # Blank frame
                else:
                    # Process the frame
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    processed_frame = cv2.resize(processed_frame, (320, 240))

                    # Add label for priority camera
                    if i == priority_camera_index:
                        cv2.putText(processed_frame, f"Priority Camera {i}", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(processed_frame, f"Camera {i} Frame", (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                frames.append(processed_frame)
            else:
                # Add blank frames for skipped intervals
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
    camera_indices = [0, 1]  # Indices for the two cameras
    priority_camera_index = 0  # Camera 0 is the priority camera

    print("Starting webcam stream processing with a priority camera...")

    # Start the camera details thread
    details_thread = threading.Thread(
        target=display_camera_details,
        args=(camera_indices, priority_camera_index, 30, 1)
    )
    details_thread.daemon = True  # This makes the thread exit when the main program ends
    details_thread.start()

    # Start processing webcam streams
    process_webcam_streams(camera_indices, priority_camera_index=priority_camera_index, priority_fps=30, other_fps=1)


if __name__ == "__main__":
    main()
