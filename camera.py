import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from pathlib import Path


# Tiff start add head movement --- Head movement calibration ---
calib_frames_needed = 30
calib_count = 0
nose_x_samples = []
nose_y_samples = []
shoulder_mid_x_samples = []
shoulder_mid_y_samples = []
nose_minus_eye_y_samples = []
# thresholds (tweak)
TURN_THRESH_X = 0.03
LOOK_THRESH_Y = 0.03
smooth_x = 0.0
smooth_y = 0.0
ALPHA = 0.85
# tiff end

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

POSE_CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32)]

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)        
        action = "None"
        head_action = "No pose"
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            for connection in POSE_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                cv2.line(frame, (int(start.x*w), int(start.y*h)), (int(end.x*w), int(end.y*h)), (0,255,0), 2)
            
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 3, (0,0,255), -1)
            
            if landmarks[15].y < landmarks[11].y and landmarks[16].y < landmarks[12].y:
                action = "Gangnam Style"
            elif abs(landmarks[25].y - landmarks[23].y) > 0.15 or abs(landmarks[26].y - landmarks[24].y) > 0.15:
                action = "Running in Place"

            head_action = "Forward"

            nose = landmarks[0]
            left_eye = landmarks[5]
            right_eye = landmarks[2]
            eye_mid_y = (left_eye.y + right_eye.y) / 2.0

            ls = landmarks[11]
            rs = landmarks[12]

            shoulder_mid_x = (ls.x + rs.x) / 2.0
            shoulder_mid_y = (ls.y + rs.y) / 2.0

            # calibration
            if calib_count < calib_frames_needed:
                nose_x_samples.append(nose.x)
                nose_y_samples.append(nose.y)
                shoulder_mid_x_samples.append(shoulder_mid_x)
                shoulder_mid_y_samples.append(shoulder_mid_y)

                left_eye = landmarks[5]
                right_eye = landmarks[2]
                eye_mid_y = (left_eye.y + right_eye.y) / 2.0
                nose_minus_eye_y_samples.append(nose.y - eye_mid_y)

                calib_count += 1
                head_action = "Calibrating"
            else:
                base_nose_x = np.mean(nose_x_samples)
                base_nose_y = np.mean(nose_y_samples)
                base_sh_x = np.mean(shoulder_mid_x_samples)
                base_sh_y = np.mean(shoulder_mid_y_samples)

                rel_x = (nose.x - shoulder_mid_x) - (base_nose_x - base_sh_x)
                left_eye = landmarks[5]
                right_eye = landmarks[2]
                eye_mid_y = (left_eye.y + right_eye.y) / 2.0

                base_nose_minus_eye_y = float(np.mean(nose_minus_eye_y_samples))
                rel_y = (nose.y - eye_mid_y) - base_nose_minus_eye_y
                smooth_x = ALPHA * smooth_x + (1 - ALPHA) * rel_x
                smooth_y = ALPHA * smooth_y + (1 - ALPHA) * rel_y

                if smooth_y < -LOOK_THRESH_Y:
                    head_action = "Look Up"
                elif smooth_y > LOOK_THRESH_Y:
                    head_action = "Look Down"
                elif smooth_x > TURN_THRESH_X:
                    head_action = "Turn Right"
                elif smooth_x < -TURN_THRESH_X:
                    head_action = "Turn Left"
                else:
                    head_action = "Forward"
        
        
        cv2.rectangle(frame, (5, 5), (420, 110), (0, 0, 0), -1)  # black background box
        cv2.putText(frame, f"Action: {action}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Head: {head_action}", (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
