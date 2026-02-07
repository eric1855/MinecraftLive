import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import pyautogui
import math
import time
from key_emulator import set_key, release_all, tap

# =========================== CONFIGURATION ===========================
CONFIG = {
    # --- Body / Running Config ---
    "running_mode": "arms",  # "arms", "legs", or "arms_or_legs"
    "arm_left_wrist": 15, "arm_left_shoulder": 11,
    "arm_right_wrist": 16, "arm_right_shoulder": 12,
    "running_arms_require_both": False,
    "arm_raise_margin": 0.0,
    "leg_left_hip": 23, "leg_left_knee": 25,
    "leg_right_hip": 24, "leg_right_knee": 26,
    "leg_knee_hip_y_diff_threshold": 0.15,
    "running_confirm_frames": 2,
    "running_release_frames": 3,
    "running_key": "w",
    "tpose_key": "e",
    "jump_key": "space",

    # --- Head Tracking Config ---
    "head_turn_thresh_x": 0.03,
    "head_look_thresh_y": 0.03,
    "head_smooth_alpha": 0.85,
    "calib_frames_needed": 30,
}

# =========================== INITIALIZATION ===========================
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Using 'full' model is recommended for better head tracking accuracy, 
# but you can switch to 'lite' if it lags.
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screen_width, screen_height = pyautogui.size()

# --- State Variables ---
# Running Smoothing
_n = max(CONFIG["running_confirm_frames"], CONFIG["running_release_frames"])
_running_buffer = deque([False] * _n, maxlen=_n)

# Logic State
prev_action = None
prev_hip_y = None
hand_tracking_enabled = False
prev_tpose = False
prev_fist_state = False
hand_delta_x = 0
hand_delta_y = 0

# Head Tracking Calibration State
calib_count = 0
nose_x_samples = []
nose_y_samples = []
shoulder_mid_x_samples = []
shoulder_mid_y_samples = []
nose_minus_eye_y_samples = []

# Head Tracking Runtime State
base_nose_x = 0
base_nose_y = 0
base_sh_x = 0
base_sh_y = 0
base_nose_minus_eye_y = 0
smooth_head_x = 0.0
smooth_head_y = 0.0

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),
    (13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),
    (16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]
# =========================== HELPER FUNCTIONS ===========================

def _detect_running_raw(landmarks) -> bool:
    mode = CONFIG["running_mode"]
    cfg = CONFIG

    if mode in ("arms", "arms_or_legs"):
        left_raised = landmarks[cfg["arm_left_wrist"]].y + cfg["arm_raise_margin"] <= landmarks[cfg["arm_left_shoulder"]].y
        right_raised = landmarks[cfg["arm_right_wrist"]].y + cfg["arm_raise_margin"] <= landmarks[cfg["arm_right_shoulder"]].y
        arms_ok = (left_raised and right_raised) if cfg["running_arms_require_both"] else (left_raised or right_raised)
        if mode == "arms": return arms_ok
        if arms_ok: return True

    if mode in ("legs", "arms_or_legs"):
        ly = abs(landmarks[cfg["leg_left_knee"]].y - landmarks[cfg["leg_left_hip"]].y)
        ry = abs(landmarks[cfg["leg_right_knee"]].y - landmarks[cfg["leg_right_hip"]].y)
        if ly > cfg["leg_knee_hip_y_diff_threshold"] or ry > cfg["leg_knee_hip_y_diff_threshold"]:
            return True
    return False

def _detect_fist(hand_landmarks) -> bool:
    wrist = hand_landmarks[0]
    fingers = [(6, 8), (10, 12), (14, 16), (18, 20)] # PIP, TIP pairs
    folded_count = 0
    for pip_idx, tip_idx in fingers:
        pip = hand_landmarks[pip_idx]
        tip = hand_landmarks[tip_idx]
        dist_tip_wrist = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        dist_pip_wrist = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
        if dist_tip_wrist < dist_pip_wrist:
            folded_count += 1
    return folded_count >= 3

def _running_after_smoothing(raw_running: bool) -> bool:
    n_confirm = CONFIG["running_confirm_frames"]
    n_release = CONFIG["running_release_frames"]
    _running_buffer.append(raw_running)
    recent = list(_running_buffer)
    if sum(recent) >= n_confirm and raw_running: return True
    if sum(1 for r in recent if not r) >= n_release and not raw_running: return False
    return sum(recent) > len(recent) / 2

# =========================== MAIN LOOP ===========================

with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Use time.time for correct timestamping
        timestamp_ms = int(time.time() * 1000)
        
        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        body_action = "None"
        head_action = "No pose"
        running_raw = False
        
        # ---------------- BODY & HEAD TRACKING ----------------
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]

            # Draw Body Skeleton
            for connection in POSE_CONNECTIONS:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                cv2.line(frame, (int(start.x*w), int(start.y*h)), (int(end.x*w), int(end.y*h)), (0,255,0), 2)

            # --- 1. HEAD TRACKING LOGIC ---
            nose = landmarks[0]
            left_eye = landmarks[5]
            right_eye = landmarks[2]
            ls = landmarks[11]
            rs = landmarks[12]
            
            shoulder_mid_x = (ls.x + rs.x) / 2.0
            shoulder_mid_y = (ls.y + rs.y) / 2.0
            eye_mid_y = (left_eye.y + right_eye.y) / 2.0

            if calib_count < CONFIG["calib_frames_needed"]:
                # Calibration Phase
                nose_x_samples.append(nose.x)
                nose_y_samples.append(nose.y)
                shoulder_mid_x_samples.append(shoulder_mid_x)
                shoulder_mid_y_samples.append(shoulder_mid_y)
                nose_minus_eye_y_samples.append(nose.y - eye_mid_y)
                
                calib_count += 1
                head_action = f"Calibrating {calib_count}/{CONFIG['calib_frames_needed']}"
            else:
                # Calculate Baselines once
                if calib_count == CONFIG["calib_frames_needed"]:
                    base_nose_x = np.mean(nose_x_samples)
                    base_nose_y = np.mean(nose_y_samples)
                    base_sh_x = np.mean(shoulder_mid_x_samples)
                    base_sh_y = np.mean(shoulder_mid_y_samples)
                    base_nose_minus_eye_y = float(np.mean(nose_minus_eye_y_samples))
                    calib_count += 1 # Stop recalibrating

                # Runtime Detection
                rel_x = (nose.x - shoulder_mid_x) - (base_nose_x - base_sh_x)
                rel_y = (nose.y - eye_mid_y) - base_nose_minus_eye_y
                
                alpha = CONFIG["head_smooth_alpha"]
                smooth_head_x = alpha * smooth_head_x + (1 - alpha) * rel_x
                smooth_head_y = alpha * smooth_head_y + (1 - alpha) * rel_y

                if smooth_head_y < -CONFIG["head_look_thresh_y"]:
                    head_action = "Look Up"
                elif smooth_head_y > CONFIG["head_look_thresh_y"]:
                    head_action = "Look Down"
                elif smooth_head_x > CONFIG["head_turn_thresh_x"]:
                    head_action = "Turn Left" # Flipped due to mirror
                elif smooth_head_x < -CONFIG["head_turn_thresh_x"]:
                    head_action = "Turn Right" # Flipped due to mirror
                else:
                    head_action = "Forward"

            # --- 2. BODY ACTION LOGIC ---
            # T-Pose Detection (Arms out)
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            th_x, th_y = 0.12, 0.15
            
            arms_ext_x = (abs(left_wrist.x - ls.x) > th_x) and (abs(right_wrist.x - rs.x) > th_x)
            arms_lvl_y = abs(left_wrist.y - ls.y) < th_y and abs(right_wrist.y - rs.y) < th_y
            is_tpose = arms_ext_x and arms_lvl_y

            # Toggle Hand Tracking
            if is_tpose and not prev_tpose:
                hand_tracking_enabled = not hand_tracking_enabled
            prev_tpose = is_tpose

            # Jump Detection (Fast hip rise)
            avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
            if prev_hip_y is not None and (prev_hip_y - avg_hip_y) > 0.05:
                body_action = "Jump"
            elif is_tpose:
                body_action = "T-Pose"
            else:
                running_raw = _detect_running_raw(landmarks)
                if running_raw:
                    body_action = "Running (W)"

            prev_hip_y = avg_hip_y

        # --- KEYBOARD INPUT HANDLING ---
        running_smoothed = _running_after_smoothing(running_raw)
        set_key(CONFIG["running_key"], running_smoothed)

        if body_action == "T-Pose" and prev_action != "T-Pose":
            tap(CONFIG["tpose_key"])
        if body_action == "Jump" and prev_action != "Jump":
            tap(CONFIG["jump_key"])
        
        prev_action = body_action

        # ---------------- HAND TRACKING (MOUSE) ----------------
        if hand_tracking_enabled and hand_results.hand_landmarks:
            hand_landmarks = hand_results.hand_landmarks[0]
            
            # Draw Hand Skeleton
            for landmark in hand_landmarks:
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 5, (255,0,255), -1)

            # Fist Click
            is_fist = _detect_fist(hand_landmarks)
            if is_fist and not prev_fist_state:
                pyautogui.click()
                cv2.putText(frame, "CLICK", (int(hand_landmarks[0].x*w), int(hand_landmarks[0].y*h)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            prev_fist_state = is_fist

            # Cursor Movement (Wrist to Middle Finger Angle)
            wrist = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            dx = middle_tip.x - wrist.x
            dy = middle_tip.y - wrist.y
            
            # Sensitivity Map
            angle_x = np.clip(dx * 5, -1, 1) # Normal X (since we flipped frame)
            angle_y = np.clip(dy * 5, -1, 1)
            
            target_x = (angle_x + 1) / 2 * screen_width
            target_y = (angle_y + 1) / 2 * screen_height
            
            delta_x = (target_x - screen_width / 2) * 0.15
            delta_y = (target_y - screen_height / 2) * 0.15
            
            # Smooth Cursor
            hand_delta_x = hand_delta_x * 0.1 + delta_x * 0.9
            hand_delta_y = hand_delta_y * 0.1 + delta_y * 0.9
            
            if abs(hand_delta_x) > 0.1 or abs(hand_delta_y) > 0.1:
                pyautogui.moveRel(hand_delta_x, hand_delta_y, _pause=False)
        else:
            hand_delta_x, hand_delta_y = 0, 0

        # ---------------- UI OVERLAY ----------------
        # Black background box
        cv2.rectangle(frame, (5, 5), (450, 150), (0, 0, 0), -1)
        
        # Text Info
        color_run = (0, 255, 0) if running_smoothed else (200, 200, 200)
        color_head = (255, 255, 0)
        
        cv2.putText(frame, f"Body: {body_action}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_run, 2)
        cv2.putText(frame, f"Head: {head_action}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color_head, 2)
        
        mouse_status = "ON" if hand_tracking_enabled else "OFF (T-Pose to toggle)"
        cv2.putText(frame, f"Mouse: {mouse_status}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow('Merged Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
release_all()
cv2.destroyAllWindows()