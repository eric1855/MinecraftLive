import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import pyautogui
import math  # Added for new fist detection

from key_emulator import set_key, release_all, tap

# ============== TWEAKABLE CONFIG (adjust for your camera / pose) ==============
CONFIG = {
    # How we detect "running in place" (arms pumping vs legs)
    # "arms" = at least one wrist above its shoulder (arm raised)
    # "legs" = knee vertical offset from hip above threshold (leg bent)
    # "arms_or_legs" = either condition
    "running_mode": "arms",

    # --- Arm-based running (shaking arms up/down) ---
    # Landmarks: 11=left shoulder, 12=right, 13=left elbow, 14=right, 15=left wrist, 16=right wrist
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
    
    # Key to press when jumping
    "jump_key": "space",
}
# =============================================================================

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

cap = cv2.VideoCapture(0)
frame_count = 0
prev_action = None
prev_hip_y = None

# Smoothing buffer: True = running detected this frame (start with not-running)
_n = max(CONFIG["running_confirm_frames"], CONFIG["running_release_frames"])
_running_buffer = deque([False] * _n, maxlen=_n)

# Hand tracking toggle state
hand_tracking_enabled = False
prev_tpose = False
screen_width, screen_height = pyautogui.size()
prev_delta_x = 0
prev_delta_y = 0

# Fist detection state tracking
prev_fist_state = False  # Track fist state for single hand

# Hand tracking deltas
hand_delta_x = 0
hand_delta_y = 0

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


def _detect_fist(hand_landmarks) -> bool:
    """
    Detects if hand is making a fist by checking if fingers are folded.
    A finger is folded if the Tip is closer to the Wrist than the PIP (middle knuckle) is.
    """
    
    # 0: Wrist
    wrist = hand_landmarks[0]
    
    # Finger pairs: (PIP_Joint_Index, Tip_Index)
    # Index: 6, 8 | Middle: 10, 12 | Ring: 14, 16 | Pinky: 18, 20
    fingers = [(6, 8), (10, 12), (14, 16), (18, 20)]
    
    folded_count = 0
    
    for pip_idx, tip_idx in fingers:
        pip = hand_landmarks[pip_idx]
        tip = hand_landmarks[tip_idx]
        
        # Calculate squared Euclidean distance to avoid expensive sqrt
        # (Tip to Wrist)^2
        dist_tip_wrist = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        # (PIP to Wrist)^2
        dist_pip_wrist = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
        
        # If tip is closer to wrist than PIP is, the finger is bent/folded
        if dist_tip_wrist < dist_pip_wrist:
            folded_count += 1
            
    # We ignore the thumb because it's often tucked differently.
    # If 3 or more of the main 4 fingers are folded, it's a fist.
    return folded_count >= 3


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


with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        pose_results = pose_landmarker.detect_for_video(mp_image, frame_count)
        hand_results = hand_landmarker.detect_for_video(mp_image, frame_count)
        
        action = "None"
        running_raw = False
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]

            for connection in POSE_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                cv2.line(frame, (int(start.x*w), int(start.y*h)), (int(end.x*w), int(end.y*h)), (0,255,0), 2)

            for landmark in landmarks:
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 3, (0,0,255), -1)
            
            # T-pose detection: both arms extended horizontally
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            # T-pose thresholds
            th_x = 0.12
            th_y = 0.15
            
            # Check if arms are extended horizontally (wrists far from body, near shoulder height)
            arms_extended_x = (abs(left_wrist.x - left_shoulder.x) > th_x) and (abs(right_wrist.x - right_shoulder.x) > th_x)
            arms_level_y = abs(left_wrist.y - left_shoulder.y) < th_y and abs(right_wrist.y - right_shoulder.y) < th_y
            is_tpose = arms_extended_x and arms_level_y
            
            # Toggle hand tracking on T-pose (edge detection)
            if is_tpose and not prev_tpose:
                hand_tracking_enabled = not hand_tracking_enabled
            prev_tpose = is_tpose
            
            # Track hip movement for jump detection
            avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
            
            # Check for fast upward movement (jump)
            if prev_hip_y is not None and (prev_hip_y - avg_hip_y) > 0.05:
                action = "Jump"
            elif is_tpose:
                action = "T-Pose"
            elif abs(landmarks[25].y - landmarks[23].y) > 0.15 or abs(landmarks[26].y - landmarks[24].y) > 0.15:
                action = "Running in Place"
            else:
                running_raw = _detect_running_raw(landmarks)
                if running_raw:
                    action = "Running in Place (W held)"
            
            prev_hip_y = avg_hip_y

        # Smooth and drive key: hold W while "running in place" is detected
        running_smoothed = _running_after_smoothing(running_raw)
        set_key(CONFIG["running_key"], running_smoothed)
        
        # Tap E once when entering T-Pose
        if action == "T-Pose" and prev_action != "T-Pose":
            try:
                tap(CONFIG["tpose_key"])
            except Exception:
                pass
        
        # Tap Space when jumping
        if action == "Jump" and prev_action != "Jump":
            try:
                tap(CONFIG["jump_key"])
            except Exception:
                pass
        
        prev_action = action
        
        # Hand tracking for cursor control (only when enabled)
        if hand_tracking_enabled and hand_results.hand_landmarks:
            hand_landmarks = hand_results.hand_landmarks[0]
            
            # Draw hand skeleton
            for i, landmark in enumerate(hand_landmarks):
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 5, (255,0,255), -1)
            
            # Check for pointing gesture and click
            is_fist = _detect_fist(hand_landmarks)
            if is_fist and not prev_fist_state:
                pyautogui.click()
            prev_fist_state = is_fist
            
            # Visual feedback for fist detection
            if is_fist:
                cv2.putText(frame, "FIST DETECTED - CLICK", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Use hand angle for cursor control
            # Calculate hand direction from wrist (0) to middle finger tip (12)
            wrist = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            
            # Calculate angle from wrist to middle finger
            dx = middle_tip.x - wrist.x
            dy = middle_tip.y - wrist.y
            
            # Normalize angle to -1..1 range for both axes
            # dx: negative = pointing right, positive = pointing left (reversed)
            # dy: positive = pointing down, negative = pointing up
            angle_x = np.clip(-dx * 5, -1, 1)  # Reversed X axis, more sensitive
            angle_y = np.clip(dy * 5, -1, 1)
            
            # Map hand angle to screen position
            # angle_x: -1 = left edge, 1 = right edge
            # angle_y: -1 = top edge, 1 = bottom edge
            target_x = (angle_x + 1) / 2 * screen_width
            target_y = (angle_y + 1) / 2 * screen_height
            
            # Calculate relative movement from center
            delta_x = (target_x - screen_width / 2) * 0.15  # Increased scaling for faster response
            delta_y = (target_y - screen_height / 2) * 0.15
            
            # Smooth the deltas (minimal smoothing for quick response)
            delta_x = hand_delta_x * 0.1 + delta_x * 0.9  # Very responsive
            delta_y = hand_delta_y * 0.1 + delta_y * 0.9
            hand_delta_x = delta_x
            hand_delta_y = delta_y
            
            # Send relative movements with lower deadzone
            if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                steps = 2
                for i in range(steps):
                    pyautogui.moveRel(delta_x / steps, delta_y / steps, _pause=False)
        else:
            # Reset hand deltas when hand tracking is disabled or no hands detected
            hand_delta_x = 0
            hand_delta_y = 0

        cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if hand_tracking_enabled:
            cv2.putText(frame, "HAND TRACKING ON", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
release_all()
cv2.destroyAllWindows()