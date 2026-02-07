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
    
    # Key to press when jumping
    "jump_key": "space",
}
# =============================================================================
# ------------------ Running detection helpers ------------------

_running_hold_counter = 0
_running_release_counter = 0

def _detect_running_raw(landmarks, cfg=CONFIG) -> bool:
    """
    Returns True if "running in place" is detected this frame.
    Supports cfg["running_mode"] in {"arms", "legs", "arms_or_legs"}.
    """
    mode = cfg.get("running_mode", "arms")

    # --- arms ---
    lw = cfg["arm_left_wrist"];  ls = cfg["arm_left_shoulder"]
    rw = cfg["arm_right_wrist"]; rs = cfg["arm_right_shoulder"]
    margin = float(cfg.get("arm_raise_margin", 0.0))

    left_arm_up  = landmarks[lw].y < (landmarks[ls].y - margin)
    right_arm_up = landmarks[rw].y < (landmarks[rs].y - margin)
    if cfg.get("running_arms_require_both", False):
        arms_running = left_arm_up and right_arm_up
    else:
        arms_running = left_arm_up or right_arm_up

    # --- legs ---
    lhip = cfg["leg_left_hip"]; lknee = cfg["leg_left_knee"]
    rhip = cfg["leg_right_hip"]; rknee = cfg["leg_right_knee"]
    thr = float(cfg.get("leg_knee_hip_y_diff_threshold", 0.15))

    left_leg_bent  = abs(landmarks[lknee].y - landmarks[lhip].y) > thr
    right_leg_bent = abs(landmarks[rknee].y - landmarks[rhip].y) > thr
    legs_running = left_leg_bent or right_leg_bent

    if mode == "arms":
        return arms_running
    if mode == "legs":
        return legs_running
    return arms_running or legs_running  # arms_or_legs


def _running_after_smoothing(running_raw: bool, cfg=CONFIG) -> bool:
    """
    Debounce running signal into a stable hold/release decision.
    Returns True when we should hold W, False when we should release W.
    """
    global _running_hold_counter, _running_release_counter

    confirm_n = int(cfg.get("running_confirm_frames", 2))
    release_n = int(cfg.get("running_release_frames", 3))

    if running_raw:
        _running_hold_counter += 1
        _running_release_counter = 0
    else:
        _running_release_counter += 1
        _running_hold_counter = 0

    # Hold after enough confirmations
    if _running_hold_counter >= confirm_n:
        return True
    # Release after enough non-running frames
    if _running_release_counter >= release_n:
        return False

    # Otherwise keep previous state (simple memory)
    # If we've ever held, keep holding until release threshold
    return _running_hold_counter > 0


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
prev_action = None
prev_hip_y = None

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
            # T-pose detection: wrists extended horizontally near shoulder height
            th_x = 0.12
            th_y = 0.15
            
            # Track hip movement for jump detection
            avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
            
            # Check for fast upward movement (jump)
            if prev_hip_y is not None and (prev_hip_y - avg_hip_y) > 0.05:
                action = "Jump"
                print(f"Jump detected! Hip movement: {prev_hip_y - avg_hip_y:.4f}")
            elif (abs(landmarks[15].x - landmarks[11].x) > th_x and abs(landmarks[16].x - landmarks[12].x) > th_x and
                    abs(landmarks[15].y - landmarks[11].y) < th_y and abs(landmarks[16].y - landmarks[12].y) < th_y):
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
            print(f"Tapping space key (CONFIG['jump_key']={CONFIG['jump_key']})")
            try:
                tap(CONFIG["jump_key"])
            except Exception as e:
                pass
        
        prev_action = action
        
        
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
