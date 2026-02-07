import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from key_emulator import set_key, release_all

# =========================== CONFIG ===========================
CONFIG = {
    "running_key": "w",

    # ---- SINGLE horizontal zone line per hand ----
    "left_up": 0.50,
    "right_up": 0.50,

    # Timing for start/stop detection
    "start_switch_window_s": 1.0,
    "stop_no_switch_s": 2.0,

    # ---- NEW: No-movement stop system ----
    "no_move_timeout_s": 0.5,

    # How much wrist movement is considered "still"
    # This is in normalized coordinates (0–1)
    # Increase if too sensitive, decrease if too lenient
    "no_move_threshold": 0.01,

    "draw_zone_lines": True,
    "debug_print": False,
}
# =============================================================

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
frame_count = 0

# State variables
_holding = False
_last_any_switch_time = None
_last_left_switch_time = None
_last_right_switch_time = None

_last_left_above = None
_last_right_above = None

# ---- NEW: movement tracking ----
_last_left_pos = None
_last_right_pos = None
_last_move_time = None

prev_action = None

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),
    (23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),
    (30,32),(27,31),(28,32)
]


def update_single_line_switch(current_above, last_above, which, now):
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


# ---- NEW FUNCTION ----
def check_no_movement(left_y, right_y, now):
    """
    Returns True if hands have been effectively still for too long
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

    # If we haven't moved for too long -> consider "no movement"
    return (now - _last_move_time) >= CONFIG["no_move_timeout_s"]


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

            for a, b in POSE_CONNECTIONS:
                sa = landmarks[a]
                sb = landmarks[b]
                cv2.line(frame, (int(sa.x*w), int(sa.y*h)),
                         (int(sb.x*w), int(sb.y*h)), (0, 255, 0), 2)

            for lm in landmarks:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 0, 255), -1)

            left_line = CONFIG["left_up"]
            right_line = CONFIG["right_up"]

            if CONFIG["draw_zone_lines"]:
                y_left = int(left_line * h)
                y_right = int(right_line * h)

                cv2.line(frame, (0, y_left), (w, y_left), (0, 255, 255), 2)
                cv2.line(frame, (0, y_right), (w, y_right), (255, 255, 0), 2)

            left_wrist_y = landmarks[15].y
            right_wrist_y = landmarks[16].y

            left_above = left_wrist_y < left_line
            right_above = right_wrist_y < right_line

            now = time.time()

            # ---- NEW: check if hands stopped moving ----
            hands_still = check_no_movement(left_wrist_y, right_wrist_y, now)

            if hands_still and _holding:
                _holding = False
                set_key(CONFIG["running_key"], False)

                if CONFIG["debug_print"]:
                    print("[debug] STOPPED due to no movement")

            _last_left_above = update_single_line_switch(
                left_above, _last_left_above, "left", now
            )

            _last_right_above = update_single_line_switch(
                right_above, _last_right_above, "right", now
            )

            start_window = CONFIG["start_switch_window_s"]

            start_condition = False
            if _last_left_switch_time and _last_right_switch_time:
                tmax = max(_last_left_switch_time, _last_right_switch_time)
                tmin = min(_last_left_switch_time, _last_right_switch_time)

                if (tmax - tmin) <= start_window and (now - tmax) <= start_window:
                    start_condition = True

            if start_condition and not _holding:
                _holding = True
                set_key(CONFIG["running_key"], True)

                if CONFIG["debug_print"]:
                    print("[debug] START running -> holding W")

            if _holding and _last_any_switch_time:
                if (now - _last_any_switch_time) >= CONFIG["stop_no_switch_s"]:
                    _holding = False
                    set_key(CONFIG["running_key"], False)

                    if CONFIG["debug_print"]:
                        print("[debug] STOP due to inactivity")

            cv2.putText(
                frame,
                f"Still:{int(hands_still)} holding:{int(_holding)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            action = "Running in Place (W held)" if _holding else "None"

        set_key(CONFIG["running_key"], _holding)

        cv2.putText(frame, action, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
release_all()
cv2.destroyAllWindows()
