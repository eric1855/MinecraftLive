import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
from key_emulator import set_key, release_all
from pynput.mouse import Button, Controller

# --- MAC MOUSE CONTROLLER ---
mouse = Controller()

# =========================== CONFIGURATION ===========================
CONFIG = {
    "running_key": "w",
    "jump_key": "space",
    "inventory_key": "e",
    
    # --- ZONES ---
    "run_start_offset": 0.15,    # Chest Level
    "run_stop_offset": -0.10,    # Waist Level
    "click_trigger_offset": 0.02, 
    "hold_threshold_s": 0.5,       
    
    # --- Sensitivity ---
    "jump_threshold": 0.06,        
    "tpose_extension": 0.12,  
    "tpose_vertical_tol": 0.20, 
    
    # --- RUNNING TIMING (TIGHTENED) ---
    "switches_required": 1,
    
    # CHANGED: You must pump both hands within 0.3s to start. 
    # This prevents accidental running when just reaching for clicks.
    "start_switch_window_s": 0.3,  
    
    # CHANGED: Reset counter faster so old movements don't linger
    "switch_count_reset_s": 0.5,
}
# ====================================================================

# Mouse Helper Functions
def force_click(button='left'):
    btn = Button.left if button == 'left' else Button.right
    mouse.press(btn)
    time.sleep(0.05)
    mouse.release(btn)

def force_down(button='left'):
    btn = Button.left if button == 'left' else Button.right
    mouse.press(btn)

def force_up(button='left'):
    btn = Button.left if button == 'left' else Button.right
    mouse.release(btn)

# Mediapipe Setup
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

# Global State
prev_tpose = False

# Running State
_running_holding = False
_last_any_switch_time = None
_last_left_switch_time = None
_last_right_switch_time = None
_last_left_above = None
_last_right_above = None
_left_switch_count = 0
_right_switch_count = 0
_insta_stop_until = 0.0

# Jump State
prev_hip_y = None
_jump_timer = 0.0
_is_jumping = False

# Click State
_prev_left_raised = False
_prev_right_raised = False
_left_raise_start_time = 0.0
_right_raise_start_time = 0.0
_left_is_dragging = False
_right_is_dragging = False
_left_feedback_text = ""
_left_feedback_timer = 0.0
_right_feedback_text = ""
_right_feedback_timer = 0.0

# Feedback Overlay State
_overlay_msg = ""
_overlay_timer = 0.0

# Helpers
def _update_switch(curr, last, which, now):
    global _last_left_switch_time, _last_right_switch_time, _last_any_switch_time
    global _left_switch_count, _right_switch_count
    if last is None: return curr
    if curr != last:
        if which == "left":
            _last_left_switch_time = now
            _left_switch_count += 1
        else:
            _last_right_switch_time = now
            _right_switch_count += 1
        _last_any_switch_time = now
    return curr

def _safe_set_key(key, pressed, now):
    if pressed and now < _insta_stop_until: return
    try: set_key(key, pressed)
    except: pass

# =========================== MAIN LOOP ===========================
with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:

    print("SYSTEM READY.")
    print("  - START RUN: SNAP both hands up (Chest) quickly!")
    print("  - STOP RUN: Drop hands below Waist.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w_px = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        now = time.time()
        
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            
            # Key Landmarks
            left_eye, right_eye = lm[5], lm[2]
            ls, rs = lm[11], lm[12]
            lw, rw = lm[15], lm[16]
            lh, rh = lm[23], lm[24] # Hips
            
            # --- CALCULATE THRESHOLDS ---
            start_thresh_y = ((ls.y + rs.y) / 2.0) + CONFIG["run_start_offset"]
            stop_thresh_y = ((lh.y + rh.y) / 2.0) + CONFIG["run_stop_offset"]
            eye_level_y = (left_eye.y + right_eye.y) / 2.0
            click_thresh_y = eye_level_y - CONFIG["click_trigger_offset"]

            # --- DRAW LINES ---
            y_start_px = int(start_thresh_y * h)
            cv2.line(frame, (0, y_start_px), (w_px, y_start_px), (255, 200, 0), 2)
            cv2.putText(frame, "START (SNAP UP)", (10, y_start_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            y_stop_px = int(stop_thresh_y * h)
            cv2.line(frame, (0, y_stop_px), (w_px, y_stop_px), (0, 255, 0), 2)
            cv2.putText(frame, "STOP (DROP)", (10, y_stop_px + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            y_click_px = int(click_thresh_y * h)
            cv2.line(frame, (0, y_click_px), (w_px, y_click_px), (0, 0, 255), 2)

            # --- RUNNING LOGIC ---
            l_above_start = lw.y < start_thresh_y
            r_above_start = rw.y < start_thresh_y
            l_below_stop = lw.y > stop_thresh_y
            r_below_stop = rw.y > stop_thresh_y
            
            _last_left_above = _update_switch(l_above_start, _last_left_above, "left", now)
            _last_right_above = _update_switch(r_above_start, _last_right_above, "right", now)
            
            # Reset counters if too slow (now 0.5s)
            if _last_any_switch_time and (now - _last_any_switch_time) > CONFIG["switch_count_reset_s"]:
                _left_switch_count = _right_switch_count = 0
            
            # START Condition
            req = CONFIG["switches_required"]
            if _left_switch_count >= req and _right_switch_count >= req:
                t_max = max(_last_left_switch_time, _last_right_switch_time)
                t_min = min(_last_left_switch_time, _last_right_switch_time)
                # STRICT WINDOW (0.3s)
                if (t_max - t_min) <= CONFIG["start_switch_window_s"] and (now - t_max) <= CONFIG["start_switch_window_s"]:
                    if now >= _insta_stop_until and not _running_holding:
                        _running_holding = True
                        _left_switch_count = _right_switch_count = 0 
            
            # STOP Condition
            if l_below_stop and r_below_stop:
                _running_holding = False

            # --- CLICK LOGIC ---
            l_click_active = lw.y < click_thresh_y
            r_click_active = rw.y < click_thresh_y
            hold_time = CONFIG["hold_threshold_s"]

            if l_click_active:
                if not _prev_left_raised: _left_raise_start_time = now
                dur = now - _left_raise_start_time
                if dur > hold_time and not _left_is_dragging:
                    if now >= _insta_stop_until:
                        force_down('left')
                        _left_is_dragging = True
                        _left_feedback_text = "DRAGGING"
            else:
                if _prev_left_raised: 
                    if _left_is_dragging:
                        force_up('left')
                        _left_is_dragging = False
                        _left_feedback_text = "RELEASED"
                        _left_feedback_timer = now + 1.0
                    else:
                        force_click('left')
                        _left_feedback_text = "CLICK!"
                        _left_feedback_timer = now + 1.0

            if r_click_active:
                if not _prev_right_raised: _right_raise_start_time = now
                dur = now - _right_raise_start_time
                if dur > hold_time and not _right_is_dragging:
                    if now >= _insta_stop_until:
                        force_down('right')
                        _right_is_dragging = True
                        _right_feedback_text = "DRAGGING"
            else:
                if _prev_right_raised:
                    if _right_is_dragging:
                        force_up('right')
                        _right_is_dragging = False
                        _right_feedback_text = "RELEASED"
                        _right_feedback_timer = now + 1.0
                    else:
                        force_click('right')
                        _right_feedback_text = "CLICK!"
                        _right_feedback_timer = now + 1.0
            
            _prev_left_raised = l_click_active
            _prev_right_raised = r_click_active

            # --- T-POSE (PRESS E) ---
            ext_thresh = CONFIG["tpose_extension"]
            vert_thresh = CONFIG["tpose_vertical_tol"]
            arms_wide = (abs(lw.x - ls.x) > ext_thresh) and (abs(rw.x - rs.x) > ext_thresh)
            arms_level = (abs(lw.y - ls.y) < vert_thresh) and (abs(rw.y - rs.y) < vert_thresh)
            is_tpose = arms_wide and arms_level
            
            if is_tpose:
                cv2.rectangle(frame, (100, 100), (w_px-100, h-100), (0, 255, 0), 4)
                cv2.putText(frame, "T-POSE", (int(w_px/2)-60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not prev_tpose:
                    _safe_set_key(CONFIG["inventory_key"], True, now)
                    time.sleep(0.05)
                    _safe_set_key(CONFIG["inventory_key"], False, now)
                    _overlay_msg = "INVENTORY (E)"
                    _overlay_timer = now + 1.5
            
            prev_tpose = is_tpose
            
            # --- JUMP LOGIC ---
            avg_hip_y = (lm[23].y + lm[24].y) / 2
            if _is_jumping:
                if now > _jump_timer:
                    _safe_set_key(CONFIG["jump_key"], False, now)
                    _is_jumping = False
            else:
                if prev_hip_y is not None:
                    diff = prev_hip_y - avg_hip_y
                    if diff > CONFIG["jump_threshold"]:
                        _safe_set_key(CONFIG["jump_key"], True, now)
                        _jump_timer = now + 0.1
                        _is_jumping = True
                        _overlay_msg = "JUMP!"
                        _overlay_timer = now + 1.0
            
            prev_hip_y = avg_hip_y

            # --- VISUALS ---
            if now < _left_feedback_timer:
                cv2.putText(frame, _left_feedback_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if now < _right_feedback_timer:
                cv2.putText(frame, _right_feedback_text, (w_px-200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if now < _overlay_timer:
                cv2.putText(frame, _overlay_msg, (int(w_px/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

            if l_click_active and not _left_is_dragging:
                pct = min(1.0, (now - _left_raise_start_time)/hold_time)
                cv2.rectangle(frame, (50, 200), (50+int(100*pct), 210), (0,255,255), -1)
            if r_click_active and not _right_is_dragging:
                pct = min(1.0, (now - _right_raise_start_time)/hold_time)
                cv2.rectangle(frame, (w_px-150, 200), (w_px-150+int(100*pct), 210), (0,255,255), -1)

        _safe_set_key(CONFIG["running_key"], _running_holding, now)
        
        status = "RUNNING" if _running_holding else "STOPPED"
        col = (0, 255, 0) if _running_holding else (0, 0, 255)
        cv2.putText(frame, f"Mode: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        
        cv2.imshow('Action Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
release_all()
cv2.destroyAllWindows()