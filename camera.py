import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque

from key_emulator import set_key, release_all, tap

# ============== TWEAKABLE CONFIG (adjust for your camera / pose) ==============
CONFIG = {
    # How we detect "running in place" (arms pumping vs legs)
    # "arms" = at least one wrist above its shoulder (arm raised)
    # "legs" = knee vertical offset from hip above threshold (leg bent)
    # "arms_or_legs" = either condition
    "running_mode": "arms",

    # --- Arm-based running (shaking arms up/down) ---
    # Landmark indices: 11=left shoulder, 12=right, 13=left elbow, 14=right, 15=left wrist, 16=right wrist
    "arm_left_wrist": 15,
    "arm_left_shoulder": 11,
    "arm_right_wrist": 16,
    "arm_right_shoulder": 12,
    # Require BOTH arms to be "raised" (wrist above shoulder), or just one
    "running_arms_require_both": False,
    # Optional: wrist must be this much above shoulder (in normalized y; smaller y = higher)
    # Set to 0 to use any "wrist above shoulder"
    "arm_raise_margin": 0.0,

    # --- Leg-based running (current knee/hip check) ---
    # Landmarks: 23=left hip, 24=right hip, 25=left knee, 26=right knee
    "leg_left_hip": 23,
    "leg_left_knee": 25,
    "leg_right_hip": 24,
    "leg_right_knee": 26,
    "leg_knee_hip_y_diff_threshold": 0.15,

    # --- Smoothing (reduces jitter, makes W hold more stable) ---
    # Number of consecutive frames that must show "running" before we press W
    "running_confirm_frames": 2,
    # Number of consecutive frames that must show "not running" before we release W
    "running_release_frames": 3,

    # Key to hold while "running in place" is detected
    "running_key": "w",
    
    # Key to press when T-Pose is detected
    "tpose_key": "e",
}
# =============================================================================

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture(0)
frame_count = 0
prev_action = None

# Smoothing buffer: True = running detected this frame (start with not-running)
_n = max(CONFIG["running_confirm_frames"], CONFIG["running_release_frames"])
_running_buffer = deque([False] * _n, maxlen=_n)

POSE_CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32)]


def _detect_running_raw(landmarks) -> bool:
    """True if current frame looks like running (no smoothing)."""
    mode = CONFIG["running_mode"]
    cfg = CONFIG

    if mode in ("arms", "arms_or_legs"):
        # Arm pump: wrist above shoulder (smaller y = higher in image)
        left_raised = landmarks[cfg["arm_left_wrist"]].y + cfg["arm_raise_margin"] <= landmarks[cfg["arm_left_shoulder"]].y
        right_raised = landmarks[cfg["arm_right_wrist"]].y + cfg["arm_raise_margin"] <= landmarks[cfg["arm_right_shoulder"]].y
        arms_ok = (left_raised and right_raised) if cfg["running_arms_require_both"] else (left_raised or right_raised)
        if mode == "arms":
            return arms_ok
        if arms_ok:
            return True

    if mode in ("legs", "arms_or_legs"):
        ly = abs(landmarks[cfg["leg_left_knee"]].y - landmarks[cfg["leg_left_hip"]].y)
        ry = abs(landmarks[cfg["leg_right_knee"]].y - landmarks[cfg["leg_right_hip"]].y)
        if ly > cfg["leg_knee_hip_y_diff_threshold"] or ry > cfg["leg_knee_hip_y_diff_threshold"]:
            return True

    return False


def _running_after_smoothing(raw_running: bool) -> bool:
    """Apply confirm/release frame counts so W doesn't flicker."""
    n_confirm = CONFIG["running_confirm_frames"]
    n_release = CONFIG["running_release_frames"]
    _running_buffer.append(raw_running)
    recent = list(_running_buffer)
    if sum(recent) >= n_confirm and raw_running:
        return True
    if sum(1 for r in recent if not r) >= n_release and not raw_running:
        return False
    # Keep previous state when ambiguous: prefer "still running" if we had enough Trues lately
    return sum(recent) > len(recent) / 2


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
        running_raw = False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]

            for connection in POSE_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                cv2.line(frame, (int(start.x*w), int(start.y*h)), (int(end.x*w), int(end.y*h)), (0,255,0), 2)

            for landmark in landmarks:
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 3, (0,0,255), -1)
            
            # T-pose detection: wrists extended horizontally near shoulder height
            th_x = 0.12
            th_y = 0.15

            if (abs(landmarks[15].x - landmarks[11].x) > th_x and abs(landmarks[16].x - landmarks[12].x) > th_x and
                    abs(landmarks[15].y - landmarks[11].y) < th_y and abs(landmarks[16].y - landmarks[12].y) < th_y):
                action = "T-Pose"
            elif abs(landmarks[25].y - landmarks[23].y) > 0.15 or abs(landmarks[26].y - landmarks[24].y) > 0.15:
                action = "Running in Place"
            else:
                running_raw = _detect_running_raw(landmarks)
                if running_raw:
                    action = "Running in Place (W held)"

        # Smooth and drive key: hold W while "running in place" is detected
        running_smoothed = _running_after_smoothing(running_raw)
        set_key(CONFIG["running_key"], running_smoothed)
        
        # Tap E once when entering T-Pose
        if action == "T-Pose" and prev_action != "T-Pose":
            try:
                tap(CONFIG["tpose_key"])
            except Exception:
                pass
        
        prev_action = action

        cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q')
            break

cap.release()
release_all()
cv2.destroyAllWindows()
