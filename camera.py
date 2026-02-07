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
    # --- Running via SINGLE-LINE wrist crossings (NEW) ---
    "running_key": "w",
    "left_up": 0.50,   # normalized Y threshold for LEFT wrist (15)
    "right_up": 0.50,  # normalized Y threshold for RIGHT wrist (16)

    "start_switch_window_s": 1.0,  # both hands must cross within this window to start running
    "stop_no_switch_s": 2.0,       # stop running if no crossings for this long

    # Stop if BOTH hands have effectively no movement for this long
    "no_move_timeout_s": 0.5,
    # "Still" threshold (normalized delta in wrist y). Increase if too sensitive.
    "no_move_threshold": 0.01,

    # --- Body / legacy running config (kept for reference, but NOT used for W anymore) ---
    "running_mode": "arms",  # (unused for W unless you re-enable fallback)
    "arm_left_wrist": 15, "arm_left_shoulder": 11,
    "arm_right_wrist": 16, "arm_right_shoulder": 12,
    "running_arms_require_both": False,
    "arm_raise_margin": 0.0,
    "leg_left_hip": 23, "leg_left_knee": 25,
    "leg_right_hip": 24, "leg_right_knee": 26,
    "leg_knee_hip_y_diff_threshold": 0.15,
    "running_confirm_frames": 2,
    "running_release_frames": 3,

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
hand_delta_x = 0.0
hand_delta_y = 0.0

# Head Tracking Calibration State
calib_count = 0
nose_x_samples = []
nose_y_samples = []
shoulder_mid_x_samples = []
shoulder_mid_y_samples = []
nose_minus_eye_y_samples = []

# Head Tracking Runtime State
base_nose_x = 0.0
base_nose_y = 0.0
base_sh_x = 0.0
base_sh_y = 0.0
base_nose_minus_eye_y = 0.0
smooth_head_x = 0.0
smooth_head_y = 0.0

# ---- NEW: SINGLE-LINE RUN STATE ----
_running_holding = False
_last_any_switch_time = None
_last_left_switch_time = None
_last_right_switch_time = None
_last_left_above = None
_last_right_above = None

# ---- NEW: movement tracking for "still hands stop" ----
_last_left_pos = None
_last_right_pos = None
_last_move_time = None

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
    """
    Switch occurs when we CROSS the single line (above <-> below).
    """
    global _last_left_switch_time, _last_right_switch_time, _last_any_switch_time

    if last_above is None:
        return current_above

    if current_above != last_above:
        if which == "left":
            _last_left_switch_time = now
        else:
            _last_right_switch_time = now

        _last_any_switch_time = now

        if CONFIG["debug_print"]:
            print(f"[debug] {which} crossed line -> switch at {now:.3f}")

    return current_above


def _check_no_movement(left_y, right_y, now) -> bool:
    """
    Returns True if both hands have been effectively still for no_move_timeout_s.
    Uses normalized y delta with no_move_threshold.
    """
    global _last_left_pos, _last_right_pos, _last_move_time

    threshold = CONFIG["no_move_threshold"]

    if _last_left_pos is None:
        _last_left_pos = left_y
        _last_right_pos = right_y
        _last_move_time = now
        return False

    left_diff = abs(left_y - _last_left_pos)
    right_diff = abs(right_y - _last_right_pos)

    # If either hand moved beyond threshold -> update last move time
    if left_diff > threshold or right_diff > threshold:
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

            # Draw Body Skeleton
            for a, b in POSE_CONNECTIONS:
                sa = landmarks[a]
                sb = landmarks[b]
                cv2.line(frame, (int(sa.x*w), int(sa.y*h)), (int(sb.x*w), int(sb.y*h)), (0,255,0), 2)

            # --- 1) HEAD TRACKING ---
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

                if smooth_head_y < -CONFIG["head_look_thresh_y"]:
                    head_action = "Look Up"
                elif smooth_head_y > CONFIG["head_look_thresh_y"]:
                    head_action = "Look Down"
                elif smooth_head_x > CONFIG["head_turn_thresh_x"]:
                    head_action = "Turn Left"   # flipped due to mirror
                elif smooth_head_x < -CONFIG["head_turn_thresh_x"]:
                    head_action = "Turn Right"  # flipped due to mirror
                else:
                    head_action = "Forward"

            # --- 2) BODY ACTIONS (T-pose / Jump) ---
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            th_x, th_y = 0.12, 0.15

            arms_ext_x = (abs(left_wrist.x - ls.x) > th_x) and (abs(right_wrist.x - rs.x) > th_x)
            arms_lvl_y = abs(left_wrist.y - ls.y) < th_y and abs(right_wrist.y - rs.y) < th_y
            is_tpose = arms_ext_x and arms_lvl_y

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
                body_action = "None"
            prev_hip_y = avg_hip_y

            # --- 3) NEW RUNNING VIA SINGLE-LINE CROSSINGS ---
            now = time.time()

            left_line = CONFIG["left_up"]
            right_line = CONFIG["right_up"]

            # Draw ONLY the upper lines
            if CONFIG["draw_zone_lines"]:
                y_left = int(left_line * h)
                y_right = int(right_line * h)
                cv2.line(frame, (0, y_left), (w, y_left), (0, 255, 255), 2)
                cv2.line(frame, (0, y_right), (w, y_right), (255, 255, 0), 2)

            left_wrist_y = left_wrist.y
            right_wrist_y = right_wrist.y

            left_above = left_wrist_y < left_line
            right_above = right_wrist_y < right_line

            # Stop if both hands are still for 0.5s
            hands_still = _check_no_movement(left_wrist_y, right_wrist_y, now)
            if hands_still and _running_holding:
                _running_holding = False
                if CONFIG["debug_print"]:
                    print("[debug] STOP running due to no movement")

            # Detect crossings
            _last_left_above = _update_single_line_switch(left_above, _last_left_above, "left", now)
            _last_right_above = _update_single_line_switch(right_above, _last_right_above, "right", now)

            # Start condition: both hands crossed within window
            start_window = CONFIG["start_switch_window_s"]
            start_condition = False
            if _last_left_switch_time is not None and _last_right_switch_time is not None:
                tmax = max(_last_left_switch_time, _last_right_switch_time)
                tmin = min(_last_left_switch_time, _last_right_switch_time)
                if (tmax - tmin) <= start_window and (now - tmax) <= start_window:
                    start_condition = True

            if start_condition and not _running_holding:
                _running_holding = True
                if CONFIG["debug_print"]:
                    print("[debug] START running (two-hand cross window satisfied)")

            # Stop condition: no crossings for too long
            if _running_holding and _last_any_switch_time is not None:
                if (now - _last_any_switch_time) >= CONFIG["stop_no_switch_s"]:
                    _running_holding = False
                    if CONFIG["debug_print"]:
                        print("[debug] STOP running due to no crossings timeout")

            # If running, override body_action label
            if _running_holding and body_action == "None":
                body_action = "Running (W)"

            # Debug overlay
            cv2.putText(
                frame,
                f"RunHold:{int(_running_holding)} Still:{int(hands_still)}",
                (15, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ---------------- KEYBOARD INPUT HANDLING ----------------
        # IMPORTANT: W is driven ONLY by _running_holding now
        set_key(CONFIG["running_key"], _running_holding)

        if body_action == "T-Pose" and prev_action != "T-Pose":
            tap(CONFIG["tpose_key"])
        if body_action == "Jump" and prev_action != "Jump":
            tap(CONFIG["jump_key"])

        prev_action = body_action

        # ---------------- HAND TRACKING (MOUSE) ----------------
        if hand_tracking_enabled and hand_results.hand_landmarks:
            hand_landmarks = hand_results.hand_landmarks[0]

            for landmark in hand_landmarks:
                cv2.circle(frame, (int(landmark.x*w), int(landmark.y*h)), 5, (255,0,255), -1)

            is_fist = _detect_fist(hand_landmarks)
            if is_fist and not prev_fist_state:
                pyautogui.click()
                cv2.putText(
                    frame, "CLICK",
                    (int(hand_landmarks[0].x*w), int(hand_landmarks[0].y*h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
                )
            prev_fist_state = is_fist

            wrist = hand_landmarks[0]
            middle_tip = hand_landmarks[12]
            dx = middle_tip.x - wrist.x
            dy = middle_tip.y - wrist.y

            angle_x = np.clip(dx * 5, -1, 1)
            angle_y = np.clip(dy * 5, -1, 1)

            target_x = (angle_x + 1) / 2 * screen_width
            target_y = (angle_y + 1) / 2 * screen_height

            delta_x = (target_x - screen_width / 2) * 0.15
            delta_y = (target_y - screen_height / 2) * 0.15

            hand_delta_x = hand_delta_x * 0.1 + delta_x * 0.9
            hand_delta_y = hand_delta_y * 0.1 + delta_y * 0.9

            if abs(hand_delta_x) > 0.1 or abs(hand_delta_y) > 0.1:
                pyautogui.moveRel(hand_delta_x, hand_delta_y, _pause=False)
        else:
            hand_delta_x, hand_delta_y = 0.0, 0.0

        # ---------------- UI OVERLAY ----------------
        cv2.rectangle(frame, (5, 5), (520, 190), (0, 0, 0), -1)

        color_run = (0, 255, 0) if _running_holding else (200, 200, 200)
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
