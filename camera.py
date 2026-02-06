import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture(0)
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
        
        results = landmarker.detect_for_video(mp_image, frame_count)
        
        action = "None"
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
        
        cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
