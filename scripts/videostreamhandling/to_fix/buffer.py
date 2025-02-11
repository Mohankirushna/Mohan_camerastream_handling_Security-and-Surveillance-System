import cv2  # type: ignore
import queue
import threading
import time

class MultiCameraBuffer:
    """Handles multiple camera streams with a single background thread."""

    def __init__(self, sources, max_buffer_size=10):
        """
        sources: List of camera indices or video file paths.
        max_buffer_size: Maximum number of frames to store per camera.
        """
        self.sources = sources
        self.max_buffer_size = max_buffer_size
        self.buffers = {src: queue.Queue(maxsize=max_buffer_size) for src in sources}
        self.caps = {src: cv2.VideoCapture(src) for src in sources}
        self.stopped = False

        # Ensure cameras are opened
        for src, cap in self.caps.items():
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {src}")

        # Start one thread for capturing frames from all cameras
        self.thread = threading.Thread(target=self._update_buffers, daemon=True)
        self.thread.start()

    def _update_buffers(self):
        """Continuously captures frames from all cameras and stores them in buffers."""
        while not self.stopped:
            for src, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret:
                    continue  # Skip if frame capture fails

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
            cap.release()


# Main thread handles frame processing
if __name__ == "__main__":
    camera_sources = [0, 1]  # Change to the correct camera indices
    video_buffer = MultiCameraBuffer(sources=camera_sources, max_buffer_size=10)

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