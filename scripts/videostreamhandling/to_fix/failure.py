import cv2  # type: ignore
import queue
import threading
import time

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
if __name__ == "__main__":
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