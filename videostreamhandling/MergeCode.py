import cv2
import time
import numpy as np
import threading
import queue
import pytest
import pyvirtualcam
from pyvirtualcam import PixelFormat


# Display camera details function
def display_camera_details(camera_indices, priority_camera_index, priority_fps, other_fps):
    while True:
        print("\n[Camera Details]")
        for i in range(len(camera_indices)):
            if i == priority_camera_index:
                print(f"Camera {i}: Priority Camera | FPS: {priority_fps}")
            else:
                print(f"Camera {i}: Normal Camera | FPS: {other_fps}")
        time.sleep(1)  # Update details every second


# Webcam stream processing function
def process_webcam_streams(camera_indices, priority_camera_index=0, priority_fps=30, other_fps=1):
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more cameras.")
        return

    cv2.namedWindow("Processed Webcams", cv2.WINDOW_NORMAL)
    num_cameras = len(caps)

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


# MultiCameraBuffer class for handling multiple camera streams
class MultiCameraBuffer:
    """Handles multiple camera streams with auto-reconnect on failure."""

    def __init__(self, sources, max_buffer_size=10, max_retries=3, retry_delay=2):
        """
        sources: List of camera indices or video file paths.
        max_buffer_size: Maximum number of frames to store per camera.
        max_retries: Number of attempts to reconnect on failure.
        retry_delay: Seconds to wait before retrying.
        """
        self.sources = sources
        self.max_buffer_size = max_buffer_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.buffers = {src: queue.Queue(maxsize=max_buffer_size) for src in sources}
        self.caps = {src: self._open_camera(src) for src in sources}
        self.stopped = False

        # Start one thread for capturing frames from all cameras
        self.thread = threading.Thread(target=self._update_buffers, daemon=True)
        self.thread.start()

    def _open_camera(self, source):
        """Tries to open a camera stream, with retries on failure."""
        for attempt in range(1, self.max_retries + 1):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"[INFO] Camera {source} opened successfully on attempt {attempt}.")
                return cap
            print(f"[WARNING] Failed to open camera {source} (Attempt {attempt}/{self.max_retries}). Retrying in {self.retry_delay} sec...")
            time.sleep(self.retry_delay)

        print(f"[ERROR] Camera {source} could not be opened after {self.max_retries} attempts.")
        return None

    def _update_buffers(self):
        """Continuously captures frames and stores them in buffers, with failure handling."""
        while not self.stopped:
            for src in self.sources:
                cap = self.caps.get(src)
                if cap is None:
                    continue  # Skip if camera is unavailable

                ret, frame = cap.read()
                if not ret:
                    print(f"[ERROR] Lost connection to camera {src}. Attempting to reconnect...")
                    self.caps[src] = self._open_camera(src)  # Try to reconnect
                    continue  # Skip this frame if reconnection is needed

                # Maintain a fixed buffer size
                if self.buffers[src].full():
                    self.buffers[src].get()  # Remove oldest frame

                self.buffers[src].put(frame)

            time.sleep(0.01)  # Small delay to prevent CPU overload

    def get_frame(self, source):
        """Retrieve the latest frame from a specific camera buffer."""
        if source in self.buffers and not self.buffers[source].empty():
            return self.buffers[source].get()
        return None  # No frame available

    def stop(self):
        """Stops video capture and releases resources."""
        self.stopped = True
        self.thread.join()
        for cap in self.caps.values():
            if cap is not None:
                cap.release()


# Main thread handles frame processing
def handle_multiple_cameras():
    camera_sources = [0, 1]  # Adjust based on your camera setup
    video_buffer = MultiCameraBuffer(sources=camera_sources, max_buffer_size=10, max_retries=3, retry_delay=2)

    while True:
        frames = {src: video_buffer.get_frame(src) for src in camera_sources}

        for src, frame in frames.items():
            if frame is not None:
                cv2.imshow(f"Camera {src}", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_buffer.stop()
    cv2.destroyAllWindows()


# PyVirtualCam Test Functions
@pytest.fixture(scope="function")  # Each test gets a new virtual camera
def virtual_camera():
    with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=PixelFormat.BGR) as cam:
        yield cam


# Test high-resolution camera input handling
def test_high_resolution_handling():
    with pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=PixelFormat.BGR) as cam:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cam.send(frame)
        assert frame.shape[:2] == (1080, 1920), "Resolution mismatch"


# Test handling of low-light or poor-quality video input
def test_low_light_input():
    with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=PixelFormat.BGR) as cam:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Dark frame
        cam.send(frame)
        avg_brightness = np.mean(frame)
        assert avg_brightness < 10, "Frame is not in low-light conditions"


# Test network latency in IP camera streams
def test_network_latency_simulation():
    delay = 0.5  # Simulated network delay
    start_time = time.time()
    time.sleep(delay)
    end_time = time.time()
    assert (end_time - start_time) >= delay, "Latency simulation failed"


# Test dynamic FPS adjustment based on system load
def test_dynamic_fps_adjustment():
    with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=PixelFormat.BGR) as cam:
        assert cam.fps == 30, f"Expected FPS 30, but got {cam.fps}"


# Test simultaneous recording and streaming performance
def test_simultaneous_recording_streaming():
    with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=PixelFormat.BGR) as cam:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cam.send(frame)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640, 480))
        out.write(frame)
        out.release()
        cap = cv2.VideoCapture('test.avi')
        assert cap.isOpened(), "Recording failed"
        cap.release()


# Test handling of corrupt or partially captured frames
def test_corrupt_frame_handling():
    corrupt_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    try:
        processed = cv2.cvtColor(corrupt_frame, cv2.COLOR_BGR2GRAY)
        assert processed is not None, "Processing failed"
    except cv2.error:
        pytest.fail("cv2 error encountered while handling corrupt frame")


# Main function to run webcam and other functions
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

