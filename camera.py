import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import pyautogui
import time
from key_emulator import set_key, release_all, tap

# =========================== CONFIGURATION ===========================
CONFIG = {
    # --- Running via SINGLE-LINE wrist crossings ---
    "running_key": "w",
    "left_up": 0.50,   # normalized Y threshold for LEFT wrist
    "right_up": 0.50,  # normalized Y threshold for RIGHT wrist

    "start_switch_window_s": 1.0,  # both hands must cross within this window to start running
    "stop_no_switch_s": 2.0,       # stop running if no crossings for this long

    # Stop if BOTH hands have effectively no movement for this long
    "no_move_timeout_s": 0.5,
    "no_move_threshold": 0.01,

    # --- Actions ---
    "tpose_key": "e",
    "jump_key": "space",

    # --- Head Tracking Config ---
    "head_turn_thresh_x": 0.03,
    "head_look_thresh_y": 0.03,
    "head_smooth_alpha": 0.85,
    "calib_frames_needed": 30,

    # --- Visuals / debug ---
    "draw_zone_lines": True,
    "debug_print": False,
    
    # --- Click vs Hold Sensitivity ---
    "raise_threshold": 0.15,      # How much higher wrist must be than shoulder
    "hold_threshold_s": 0.3,      # Seconds to wait before converting a Click into a Hold
}
# ====================================================================


# =========================== INITIALIZATION ===========================
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO
)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screen_width, screen_height = pyautogui.size()

# --- State Variables ---
prev_action = None
prev_hip_y = None
hand_tracking_enabled = False
prev_tpose = False
prev_fist_state = False

# Head Tracking Calibration
calib_count = 0
nose_x_samples = []
nose_y_samples = []
shoulder_mid_x_samples = []
shoulder_mid_y_samples = []
nose_minus_eye_y_samples = []

# Head Tracking Runtime
base_nose_x = 0.0
base_nose_y = 0.0
base_sh_x = 0.0
base_sh_y = 0.0
base_nose_minus_eye_y = 0.0
smooth_head_x = 0.0
smooth_head_y = 0.0

# ---- RUNNING STATE ----
_running_holding = False
_last_any_switch_time = None
_last_left_switch_time = None
_last_right_switch_time = None
_last_left_above = None
_last_right_above = None

# ---- MOVEMENT STATE ----
_last_left_pos = None
_last_right_pos = None
_last_move_time = None
_last_left_move_time = None
_last_right_move_time = None

# ---- CLICK/HOLD STATE ----
_prev_left_raised = False
_prev_right_raised = False
_left_raise_start_time = 0.0
_right_raise_start_time = 0.0
_left_is_holding = False
_right_is_holding = False

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),
    (13,15),(15,17),(15,19),(15,21),(17,19),(12,14),(14,16),(16,18),(16,20),
    (16,22),(18,20),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]


# =========================== HELPER FUNCTIONS ===========================

def _detect_fist(hand_landmarks) -> bool:
    wrist = hand_landmarks[0]
    fingers = [(6, 8), (10, 12), (14, 16), (18, 20)]  # PIP, TIP pairs
    folded_count = 0
    for pip_idx, tip_idx in fingers:
        pip = hand_landmarks[pip_idx]
        tip = hand_landmarks[tip_idx]
        dist_tip_wrist = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        dist_pip_wrist = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
        if dist_tip_wrist < dist_pip_wrist:
            folded_count += 1
    return folded_count >= 3


def _update_single_line_switch(current_above, last_above, which, now):
    global _last_left_switch_time, _last_right_switch_time, _last_any_switch_time
    if last_above is None:
        return current_above
    if current_above != last_above:
        if which == "left":
            _last_left_switch_time = now
        else:
            _last_right_switch_time = now
        _last_any_switch_time = now
    return current_above


def _check_no_movement(left_y, right_y, now) -> bool:
    global _last_left_pos, _last_right_pos, _last_move_time, _last_left_move_time, _last_right_move_time
    threshold = CONFIG["no_move_threshold"]
    if _last_left_pos is None:
        _last_left_pos = left_y
        _last_right_pos = right_y
        _last_move_time = now
        _last_left_move_time = now
        _last_right_move_time = now
        return False

    left_diff = abs(left_y - _last_left_pos)
    right_diff = abs(right_y - _last_right_pos)

    if left_diff > threshold:
        _last_left_move_time = now
        _last_move_time = now
    if right_diff > threshold:
        _last_right_move_time = now
        _last_move_time = now

    _last_left_pos = left_y
    _last_right_pos = right_y
    return (now - _last_move_time) >= CONFIG["no_move_timeout_s"]


# =========================== MAIN LOOP ===========================

with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for intuitive interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        body_action = "None"
        head_action = "No pose"

        # ---------------- BODY & HEAD TRACKING ----------------
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]

            # Draw Skeleton
            for a, b in POSE_CONNECTIONS:
                sa = landmarks[a]
                sb = landmarks[b]
                cv2.line(frame, (int(sa.x*w), int(sa.y*h)), (int(sb.x*w), int(sb.y*h)), (0,255,0), 2)

            # --- HEAD TRACKING ---
            nose = landmarks[0]
            left_eye = landmarks[5]
            right_eye = landmarks[2]
            ls = landmarks[11]
            rs = landmarks[12]

            shoulder_mid_x = (ls.x + rs.x) / 2.0
            shoulder_mid_y = (ls.y + rs.y) / 2.0
            eye_mid_y = (left_eye.y + right_eye.y) / 2.0

            if calib_count < CONFIG["calib_frames_needed"]:
                nose_x_samples.append(nose.x)
                nose_y_samples.append(nose.y)
                shoulder_mid_x_samples.append(shoulder_mid_x)
                shoulder_mid_y_samples.append(shoulder_mid_y)
                nose_minus_eye_y_samples.append(nose.y - eye_mid_y)
                calib_count += 1
                head_action = f"Calibrating {calib_count}/{CONFIG['calib_frames_needed']}"
            else:
                if calib_count == CONFIG["calib_frames_needed"]:
                    base_nose_x = float(np.mean(nose_x_samples))
                    base_nose_y = float(np.mean(nose_y_samples))
                    base_sh_x = float(np.mean(shoulder_mid_x_samples))
                    base_sh_y = float(np.mean(shoulder_mid_y_samples))
                    base_nose_minus_eye_y = float(np.mean(nose_minus_eye_y_samples))
                    calib_count += 1

                rel_x = (nose.x - shoulder_mid_x) - (base_nose_x - base_sh_x)
                rel_y = (nose.y - eye_mid_y) - base_nose_minus_eye_y

                alpha = CONFIG["head_smooth_alpha"]
                smooth_head_x = alpha * smooth_head_x + (1 - alpha) * rel_x
                smooth_head_y = alpha * smooth_head_y + (1 - alpha) * rel_y

                if smooth_head_y < -CONFIG["head_look_thresh_y"]: head_action = "Look Up"
                elif smooth_head_y > CONFIG["head_look_thresh_y"]: head_action = "Look Down"
                elif smooth_head_x > CONFIG["head_turn_thresh_x"]: head_action = "Turn Left"
                elif smooth_head_x < -CONFIG["head_turn_thresh_x"]: head_action = "Turn Right"
                else: head_action = "Forward"

            # --- BODY ACTIONS (T-pose / Jump) ---
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            th_x, th_y = 0.12, 0.15

            arms_ext_x = (abs(left_wrist.x - ls.x) > th_x) and (abs(right_wrist.x - rs.x) > th_x)
            arms_lvl_y = abs(left_wrist.y - ls.y) < th_y and abs(right_wrist.y - rs.y) < th_y
            is_tpose = arms_ext_x and arms_lvl_y

            if is_tpose and not prev_tpose:
                hand_tracking_enabled = not hand_tracking_enabled
            prev_tpose = is_tpose 
 
            avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
            if prev_hip_y is not None and (prev_hip_y - avg_hip_y) > 0.05:
                body_action = "Jump"
            elif is_tpose:
                body_action = "T-Pose"
            prev_hip_y = avg_hip_y

            # --- RUNNING VIA WRIST CROSSINGS ---
            now = time.time()
            if CONFIG["draw_zone_lines"]:
                y_line = int(CONFIG["left_up"] * h)
                cv2.line(frame, (0, y_line), (w, y_line), (0, 255, 255), 2)

            left_wrist_y = left_wrist.y
            right_wrist_y = right_wrist.y

            left_above = left_wrist_y < CONFIG["left_up"]
            right_above = right_wrist_y < CONFIG["right_up"]

            hands_still = _check_no_movement(left_wrist_y, right_wrist_y, now)
            if hands_still and _running_holding:
                _running_holding = False

            _last_left_above = _update_single_line_switch(left_above, _last_left_above, "left", now)
            _last_right_above = _update_single_line_switch(right_above, _last_right_above, "right", now)

            start_window = CONFIG["start_switch_window_s"]
            start_condition = False
            if _last_left_switch_time is not None and _last_right_switch_time is not None:
                tmax = max(_last_left_switch_time, _last_right_switch_time)
                tmin = min(_last_left_switch_time, _last_right_switch_time)
                if (tmax - tmin) <= start_window and (now - tmax) <= start_window:
                    start_condition = True

            if start_condition and not _running_holding:
                _running_holding = True
            if _running_holding and _last_any_switch_time is not None:
                if (now - _last_any_switch_time) >= CONFIG["stop_no_switch_s"]:
                    _running_holding = False

            if _running_holding and body_action == "None":
                body_action = "Running (W)"

            cv2.putText(frame, f"Run:{int(_running_holding)} Still:{int(hands_still)}", (15, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # =================== CLICK vs HOLD LOGIC ===================
            raise_thresh = CONFIG["raise_threshold"]
            hold_time = CONFIG["hold_threshold_s"]
            
            # 1. Determine current "Raised" state
            left_is_raised = left_wrist.y < (ls.y - raise_thresh)
            right_is_raised = right_wrist.y < (rs.y - raise_thresh)

            # --- LEFT HAND LOGIC ---
            if left_is_raised:
                if not _prev_left_raised:
                    # Rising Edge: Start Timer
                    _left_raise_start_time = now
                else:
                    # Sustained: Check for Hold
                    if (now - _left_raise_start_time > hold_time) and not _left_is_holding:
                        pyautogui.mouseDown(button='left')
                        _left_is_holding = True
                        if CONFIG["debug_print"]: print("LEFT START HOLD")
            else:
                # Falling Edge (was up, now down)
                if _prev_left_raised:
                    if _left_is_holding:
                        # If we were holding, release
                        pyautogui.mouseUp(button='left')
                        _left_is_holding = False
                        if CONFIG["debug_print"]: print("LEFT END HOLD")
                    else:
                        # If we weren't holding yet, it was a quick Click
                        pyautogui.click(button='left')
                        if CONFIG["debug_print"]: print("LEFT CLICK")

            # --- RIGHT HAND LOGIC ---
            if right_is_raised:
                if not _prev_right_raised:
                    _right_raise_start_time = now
                else:
                    if (now - _right_raise_start_time > hold_time) and not _right_is_holding:
                        pyautogui.mouseDown(button='right')
                        _right_is_holding = True
                        if CONFIG["debug_print"]: print("RIGHT START HOLD")
            else:
                if _prev_right_raised:
                    if _right_is_holding:
                        pyautogui.mouseUp(button='right')
                        _right_is_holding = False
                        if CONFIG["debug_print"]: print("RIGHT END HOLD")
                    else:
                        pyautogui.click(button='right')
                        if CONFIG["debug_print"]: print("RIGHT CLICK")

            # Update State
            _prev_left_raised = left_is_raised
            _prev_right_raised = right_is_raised

            # Visual Feedback
            if _left_is_holding:
                cv2.putText(frame, "L-HOLD", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif left_is_raised:
                cv2.putText(frame, "L-UP...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
            if _right_is_holding:
                cv2.putText(frame, "R-HOLD", (w-200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif right_is_raised:
                cv2.putText(frame, "R-UP...", (w-200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # ==========================================================

        # ---------------- KEYBOARD OUTPUT ----------------
        set_key(CONFIG["running_key"], _running_holding)

        if body_action == "T-Pose" and prev_action != "T-Pose":
            tap(CONFIG["tpose_key"])
        if body_action == "Jump" and prev_action != "Jump":
            tap(CONFIG["jump_key"])

        prev_action = body_action

        # ---------------- MOUSE MOVEMENT ----------------
        if hand_tracking_enabled and hand_results.hand_landmarks:
            hand_landmarks = hand_results.hand_landmarks[0]
            for landmark in hand_landmarks:
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 5, (255,0,255), -1)

            wrist = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            dx = middle_tip.x - wrist.x
            dy = middle_tip.y - wrist.y
            
            # Mouse sensitivity
            angle_x = np.clip(dx * 5, -1, 1)
            angle_y = np.clip(dy * 5, -1, 1) 
            
            target_x = (angle_x + 1) / 2 * screen_width
            target_y = (angle_y + 1) / 2 * screen_height 
            # pyautogui.moveTo(target_x, target_y)

        mouse_status = "ON" if hand_tracking_enabled else "OFF (T-Pose to toggle)"
        cv2.putText(frame, f"Mouse: {mouse_status}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow('Action Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
release_all()
cv2.destroyAllWindows()