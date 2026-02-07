import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque

from key_emulator import set_key, release_all

# ============== TWEAKABLE CONFIG (adjust for your camera / pose) ==============
CONFIG = {
    # --- Running = hands moving in opposite directions for t > min time ---
    # Landmark indices: 15=left wrist, 16=right wrist (MediaPipe pose)
    "hand_left_wrist": 15,
    "hand_right_wrist": 16,
    # Frames of history used to compute "is this hand moving up or down?" (y in image)
    "running_velocity_window_frames": 6,
    # Minimum vertical motion (normalized y per frame) to count as "moving"
    # Increase if noise triggers; decrease if real pumps are ignored
    "running_min_velocity": 0.002,
    # Running = opposite motion for at least this many consecutive frames (t > x)
    # Higher = must pump longer before W engages; lower = quicker response
    "running_opposite_direction_min_frames": 2,

    # --- Smoothing (reduces jitter after we've already decided "running") ---
    "running_confirm_frames": 2,
    "running_release_frames": 30,
    "running_key": "w",
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

# Smoothing buffer: True = running detected this frame (start with not-running)
_n = max(CONFIG["running_confirm_frames"], CONFIG["running_release_frames"])
_running_buffer = deque([False] * _n, maxlen=_n)

# Hand motion: recent y positions for velocity (smaller y = higher in image)
_w = CONFIG["running_velocity_window_frames"] + 1
_left_hand_ys: deque = deque(maxlen=_w)
_right_hand_ys: deque = deque(maxlen=_w)
_opposite_motion_streak: int = 0

POSE_CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32)]


def _detect_running_raw(landmarks) -> bool:
    """True when both hands are moving in opposite directions for >= min frames."""
    global _opposite_motion_streak
    cfg = CONFIG
    left_y = landmarks[cfg["hand_left_wrist"]].y
    right_y = landmarks[cfg["hand_right_wrist"]].y
    _left_hand_ys.append(left_y)
    _right_hand_ys.append(right_y)

    min_vel = cfg["running_min_velocity"]
    window = cfg["running_velocity_window_frames"]
    min_frames = cfg["running_opposite_direction_min_frames"]

    if len(_left_hand_ys) <= window or len(_right_hand_ys) <= window:
        _opposite_motion_streak = 0
        return False

    # Velocity = (current - old) / span → positive = moving down in image, negative = up
    left_vel = (_left_hand_ys[-1] - _left_hand_ys[-1 - window]) / window
    right_vel = (_right_hand_ys[-1] - _right_hand_ys[-1 - window]) / window

    # Opposite directions: one moving up, one moving down, both above noise
    opposite = (
        (left_vel * right_vel < 0)
        and (abs(left_vel) >= min_vel)
        and (abs(right_vel) >= min_vel)
    )
    if opposite:
        _opposite_motion_streak = min(_opposite_motion_streak + 1, min_frames + 1)
    else:
        _opposite_motion_streak = 0

    return _opposite_motion_streak >= min_frames


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

            if landmarks[15].y < landmarks[11].y and landmarks[16].y < landmarks[12].y:
                action = "Gangnam Style"
            else:
                running_raw = _detect_running_raw(landmarks)
                if running_raw:
                    action = "Running in Place (W held)"

        # Smooth and drive key: hold W while "running in place" is detected
        running_smoothed = _running_after_smoothing(running_raw)
        set_key(CONFIG["running_key"], running_smoothed)

        cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
release_all()
cv2.destroyAllWindows()
