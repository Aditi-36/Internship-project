import cv2 
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Define pose connections for MediaPipe Pose (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

model_path = 'pose_landmarker_lite.task'

# Global variables to store latest detection results
latest_result = None
latest_output_image = None

def handle_results(result, output_image, timestamp_ms):
    global latest_result, latest_output_image
    # Store the latest result and output image for rendering in the main loop
    latest_result = result
    latest_output_image = output_image
    
cap = cv2.VideoCapture(0)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_results,  # Handing it our delivery box function
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


with PoseLandmarker.create_from_options(options) as landmarker:
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while cap.isOpened():
        res, frame = cap.read()
        if not res:
            print("Ignoring empty camera frame.")
            continue
        
        frame_count += 1
        
        # Flip the frame horizontally for a mirror view and convert BGR to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Convert OpenCV camera frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 2. Send the image and the exact time it was captured to MediaPipe
        current_timestamp_in_milliseconds = int(time.time() * 1000)
        landmarker.detect_async(mp_image, current_timestamp_in_milliseconds)
        
        # 3. Draw skeleton overlay if pose landmarks are detected
        if latest_result and latest_result.pose_landmarks:
            # Convert pose landmarks to the format needed for drawing
            h, w, c = frame.shape
            
            # Draw circles at each landmark and connections
            pose_landmarks = latest_result.pose_landmarks[0]
            
            # Draw circles at landmarks
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections (pose skeleton)
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]
                
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # 4. Calculate and display FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = 30 / elapsed_time if elapsed_time > 0 else 0
            start_time = time.time()
            print(f"FPS: {fps:.2f}")
        
        # 5. Display FPS on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 6. Performance warning if FPS drops below 15
        if fps > 0 and fps < 15:
            cv2.putText(frame, 'WARNING: FPS < 15', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the annotated frame
        cv2.imshow('Pose Detection', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Application closed.")