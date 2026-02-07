import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
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
    "run_start_offset": 0.15,     # Chest Level
    
    # CHANGE: Lowered to 0.30 (Mid-Thigh) to give more running room
    "run_stop_offset": 0.00,      
    
    "click_trigger_offset": 0.02, # Eye Level
    "hold_threshold_s": 0.5,       
    
    # --- STRICT ALIGNMENT SETTINGS ---
    "click_vertical_align": 0.15, 
    "scroll_horizontal_align": 0.15,
    "tpose_extension": 0.15,      
    "scroll_cooldown_s": 0.5,   
    
    # --- RUNNING TIMING ---
    "switches_required": 1,
    "start_switch_window_s": 0.3,  
    "switch_count_reset_s": 0.5,
    "jump_threshold": 0.06, 
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

def force_scroll(dy):
    mouse.scroll(0, dy)

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

# Scroll State
_last_scroll_time = 0.0

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
    print("  - STOP LINE LOWERED (Mid-Thigh).")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w_px = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        now = time.time()
        
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            
            # Key Landmarks
            ls, rs = lm[11], lm[12] # Shoulders
            lw, rw = lm[15], lm[16] # Wrists
            lh, rh = lm[23], lm[24] # Hips
            
            # --- ZONES ---
            start_thresh_y = ((ls.y + rs.y) / 2.0) + CONFIG["run_start_offset"]
            stop_thresh_y = ((lh.y + rh.y) / 2.0) + CONFIG["run_stop_offset"]
            
            # Click Height Limit
            click_y_limit = ((lm[5].y + lm[2].y)/2.0) - CONFIG["click_trigger_offset"]

            # --- DRAW VISUAL GUIDES (RESTORED) ---
            # 1. Start/Stop Lines
            y_start = int(start_thresh_y * h)
            y_stop = int(stop_thresh_y * h)
            cv2.line(frame, (0, y_start), (w_px, y_start), (255, 200, 0), 1)
            cv2.line(frame, (0, y_stop), (w_px, y_stop), (0, 0, 255), 2)
            cv2.putText(frame, "STOP LINE (THIGHS)", (10, y_stop - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 2. CLICK ZONES (Vertical Boxes above shoulders)
            v_tol_px = int(CONFIG["click_vertical_align"] * w_px)
            ls_px = int(ls.x * w_px)
            rs_px = int(rs.x * w_px)
            
            cv2.rectangle(frame, (ls_px - v_tol_px, 0), (ls_px + v_tol_px, int(click_y_limit*h)), (50, 50, 255), 1)
            cv2.rectangle(frame, (rs_px - v_tol_px, 0), (rs_px + v_tol_px, int(click_y_limit*h)), (50, 50, 255), 1)

            # 3. SCROLL ZONES (Horizontal Boxes beside shoulders)
            h_tol_px = int(CONFIG["scroll_horizontal_align"] * h)
            ls_y_px = int(ls.y * h)
            rs_y_px = int(rs.y * h)
            
            cv2.rectangle(frame, (0, ls_y_px - h_tol_px), (ls_px - 100, ls_y_px + h_tol_px), (255, 0, 255), 1)
            cv2.rectangle(frame, (rs_px + 100, rs_y_px - h_tol_px), (w_px, rs_y_px + h_tol_px), (255, 0, 255), 1)


            # --- GEOMETRY CHECKS (STRICT) ---
            # 1. Click Check
            l_is_vertical = abs(lw.x - ls.x) < CONFIG["click_vertical_align"]
            r_is_vertical = abs(rw.x - rs.x) < CONFIG["click_vertical_align"]
            l_click_active = l_is_vertical and (lw.y < click_y_limit)
            r_click_active = r_is_vertical and (rw.y < click_y_limit)

            # 2. Scroll Check
            l_is_horizontal = abs(lw.y - ls.y) < CONFIG["scroll_horizontal_align"]
            r_is_horizontal = abs(rw.y - rs.y) < CONFIG["scroll_horizontal_align"]
            l_is_extended = abs(lw.x - ls.x) > CONFIG["tpose_extension"]
            r_is_extended = abs(rw.x - rs.x) > CONFIG["tpose_extension"]

            is_left_scroll = l_is_horizontal and l_is_extended and not l_click_active
            is_right_scroll = r_is_horizontal and r_is_extended and not r_click_active

            # --- SCROLL & INVENTORY ---
            if is_left_scroll and is_right_scroll:
                cv2.putText(frame, "INVENTORY (E)", (int(w_px/2)-60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if not prev_tpose:
                    _safe_set_key(CONFIG["inventory_key"], True, now)
                    time.sleep(0.05)
                    _safe_set_key(CONFIG["inventory_key"], False, now)
                    _overlay_msg = "INVENTORY"
                    _overlay_timer = now + 1.5
                prev_tpose = True
            elif is_left_scroll:
                prev_tpose = False
                if now - _last_scroll_time > CONFIG["scroll_cooldown_s"]:
                    force_scroll(-1) 
                    _last_scroll_time = now
                    _overlay_msg = "SCROLL RIGHT -->"
                    _overlay_timer = now + 0.5
            elif is_right_scroll:
                prev_tpose = False
                if now - _last_scroll_time > CONFIG["scroll_cooldown_s"]:
                    force_scroll(1)
                    _last_scroll_time = now
                    _overlay_msg = "<-- SCROLL LEFT"
                    _overlay_timer = now + 0.5
            else:
                prev_tpose = False

            # --- CLICK LOGIC ---
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
                    else:
                        force_click('right')
                        _right_feedback_text = "CLICK!"
                        _right_feedback_timer = now + 1.0
            
            _prev_left_raised = l_click_active
            _prev_right_raised = r_click_active

            # --- RUNNING LOGIC (UPDATED) ---
            l_below_stop = lw.y > stop_thresh_y
            r_below_stop = rw.y > stop_thresh_y
            
            if l_below_stop or r_below_stop:
                _running_holding = False
            else:
                is_clicking_any = l_click_active or r_click_active
                if not is_clicking_any:
                    l_above_start = lw.y < start_thresh_y
                    r_above_start = rw.y < start_thresh_y
                    
                    _last_left_above = _update_switch(l_above_start, _last_left_above, "left", now)
                    _last_right_above = _update_switch(r_above_start, _last_right_above, "right", now)
                    
                    if _last_any_switch_time and (now - _last_any_switch_time) > CONFIG["switch_count_reset_s"]:
                        _left_switch_count = _right_switch_count = 0
                    
                    req = CONFIG["switches_required"]
                    if _left_switch_count >= req and _right_switch_count >= req:
                        t_max = max(_last_left_switch_time, _last_right_switch_time)
                        t_min = min(_last_left_switch_time, _last_right_switch_time)
                        if (t_max - t_min) <= CONFIG["start_switch_window_s"] and (now - t_max) <= CONFIG["start_switch_window_s"]:
                            if now >= _insta_stop_until and not _running_holding:
                                _running_holding = True
                                _left_switch_count = _right_switch_count = 0 

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

            # --- VISUAL FEEDBACK ---
            _safe_set_key(CONFIG["running_key"], _running_holding, now)
            
            # Status Bar
            status_text = "RUNNING (W)" if _running_holding else "STOPPED"
            status_color = (0, 255, 0) if _running_holding else (0, 0, 255)
            
            box_width = 300
            box_height = 60
            top_x = int(w_px/2 - box_width/2)
            cv2.rectangle(frame, (top_x, 10), (top_x + box_width, 10 + box_height), (50, 50, 50), -1) 
            cv2.rectangle(frame, (top_x, 10), (top_x + box_width, 10 + box_height), status_color, 3) 
            
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int(w_px/2 - text_size[0]/2)
            text_y = int(10 + box_height/2 + text_size[1]/2)
            cv2.putText(frame, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # Other Feedback
            if now < _left_feedback_timer:
                cv2.putText(frame, _left_feedback_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if now < _right_feedback_timer:
                cv2.putText(frame, _right_feedback_text, (w_px-200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            if now < _overlay_timer:
                cv2.putText(frame, _overlay_msg, (int(w_px/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

            # Progress Bars
            if l_click_active and not _left_is_dragging:
                pct = min(1.0, (now - _left_raise_start_time)/hold_time)
                cv2.rectangle(frame, (50, 200), (50+int(100*pct), 210), (0,255,255), -1)
            if r_click_active and not _right_is_dragging:
                pct = min(1.0, (now - _right_raise_start_time)/hold_time)
                cv2.rectangle(frame, (w_px-150, 200), (w_px-150+int(100*pct), 210), (0,255,255), -1)

        cv2.imshow('Action Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
release_all()
cv2.destroyAllWindows()