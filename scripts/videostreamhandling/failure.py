import cv2
import time

def reconnect_camera(camera_id, max_retries=3, retry_interval=5):
    """
    Attempts to reconnect a camera up to max_retries times.
    :param camera_id: The camera index or URL.
    :param max_retries: Maximum number of reconnection attempts.
    :param retry_interval: Time in seconds between attempts.
    :return: True if reconnected, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}: Trying to reconnect to camera {camera_id}")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            print(f"Camera {camera_id} reconnected successfully!")
            cap.release()
            return True
        
        print(f"Reconnection failed for camera {camera_id}. Retrying in {retry_interval} seconds...")
        cap.release()
        time.sleep(retry_interval)
    
    print(f"Camera {camera_id} could not be reconnected after {max_retries} attempts.")
    return False

def reconnect_multiple_cameras(camera_ids, max_retries=3, retry_interval=5):
    """
    Attempts to reconnect multiple cameras based on input from an external source (e.g., friend's request).
    :param camera_ids: List of camera indices or URLs provided externally.
    :param max_retries: Maximum number of reconnection attempts per camera.
    :param retry_interval: Time in seconds between attempts.
    :return: Dictionary with camera IDs as keys and reconnection status as values.
    """
    results = {}
    for camera_id in camera_ids:
        results[camera_id] = reconnect_camera(camera_id, max_retries, retry_interval)
    return results

# Example usage
if __name__ == "__main__":
    # Simulating input from a friend (external source)
    camera_ids = input("Enter faulty camera IDs separated by commas: ").split(',')
    camera_ids = [int(cam.strip()) for cam in camera_ids]  # Convert input to a list of integers
    
    statuses = reconnect_multiple_cameras(camera_ids)
    for cam_id, status in statuses.items():
        print(f"Final Status for Camera {cam_id}: {status}")
