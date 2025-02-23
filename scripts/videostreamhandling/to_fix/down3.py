def process_webcam_streams(camera_indices, priority_camera_index=0, priority_fps=30, other_fps=1):
    caps = [cv2.VideoCapture(index) for index in camera_indices]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Unable to open one or more cameras.")
        return

    cv2.namedWindow("Processed Webcams", cv2.WINDOW_NORMAL)
    num_cameras = len(caps)

    priority_interval = 1 / priority_fps
    other_interval = 1 / other_fps

    last_frame_times = [0] * num_cameras
    frame_counts = [0] * num_cameras
    actual_fps = [0.0] * num_cameras
    fps_measure_start_time = time.time()

    while True:
        frames = []
        current_time = time.time()

        for i in range(num_cameras):
            interval = priority_interval if i == priority_camera_index else other_interval

            if current_time - last_frame_times[i] >= interval:
                last_frame_times[i] = current_time

                ret, frame = caps[i].read()
                if not ret:
                    print(f"Camera {i}: Failed to read frame.")
                    processed_frame = np.zeros((240, 320), dtype=np.uint8)
                else:
                    print(f"Camera {i}: Successfully read a frame.")
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    processed_frame = cv2.resize(processed_frame, (320, 240))
                    frame_counts[i] += 1  # Increment frame count

                frames.append(processed_frame)
            else:
                frames.append(np.zeros((240, 320), dtype=np.uint8))

        combined_frame = np.hstack(frames)
        cv2.imshow("Processed Webcams", combined_frame)

        if current_time - fps_measure_start_time >= 1.0:
            for i in range(num_cameras):
                actual_fps[i] = frame_counts[i] / (current_time - fps_measure_start_time)
                print(f"Camera {i}: Actual FPS calculated as {actual_fps[i]:.2f}")
            frame_counts = [0] * num_cameras
            fps_measure_start_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
