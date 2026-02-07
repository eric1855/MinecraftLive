import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import threading
from pynput.mouse import Button, Controller
import socket
import math

# Import all key emulator functions needed
from key_emulator import set_key, release_all, key_down, key_up, tap
# import mouse_control 

# --- MAC MOUSE CONTROLLER ---
mouse = Controller()

# --- PHONE GATE (UDP) ---
PHONE_GATE_ADDR = ("127.0.0.1", 9876)
_gate_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def _set_phone_gate(enabled):
    try:
        msg = b"ENABLE" if enabled else b"DISABLE"
        _gate_sock.sendto(msg, PHONE_GATE_ADDR)
    except Exception:
        pass

# =========================== CONFIGURATION ===========================
CONFIG = {
    "running_key": "w",
    "jump_key": "space",
    "inventory_key": "e",
    
    # --- ZONES ---
    "run_start_offset": 0.15,     # Chest Level (Start)
    "run_stop_offset": 0.00,      # Just Below Hips (Stop)
    "click_trigger_offset": 0.02, # Eye Level
    
    # --- STRICT ALIGNMENT SETTINGS ---
    "click_vertical_align": 0.15, 
    "scroll_horizontal_align": 0.15,
    "tpose_extension": 0.15,      
    "scroll_cooldown_s": 0.5,   
    "hold_threshold_s": 0.5,

    # --- RUNNING TIMING ---
    "switches_required": 1,
    "start_switch_window_s": 0.5, # Tight window to prevent accidents
    "switch_count_reset_s": 1.0,
    "jump_threshold": 0.03, 

    # --- SPRINTING ---
    "sprint_speed_threshold": 0.05,  # How fast hands must move to sprint
    "walk_speed_threshold": 0.01,    # Minimum movement to stay walking
    "speed_buffer_frames": 10,       # Smooths out the "jitter" of hand movement

    # --- HEAD TRACKING ---
    "head_turn_thresh_x": 0.04,
    "head_look_thresh_y": 0.03,
    "head_smooth_alpha": 0.5,
    "calib_frames_needed": 30,
}
# ====================================================================

# --- HELPERS ---
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

def _safe_set_key(key, pressed, now):
    if pressed and now < _insta_stop_until: return
    try: set_key(key, pressed)
    except: pass

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

# --- GLOBAL STATE ---
# Running
_running_holding = False
_last_any_switch_time = None
_last_left_switch_time = 0.0
_last_right_switch_time = 0.0
_last_left_above = None
_last_right_above = None
_left_switch_count = 0
_right_switch_count = 0
_insta_stop_until = 0.0

# Jumping
prev_hip_y = None
_jump_timer = 0.0
_is_jumping = False

# Clicking
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

# Scrolling/Inventory
prev_tpose = False
_inventory_open = False
_last_scroll_time = 0.0

# Sprinting
_is_sprinting = False
_last_lw_pos = None
_last_rw_pos = None
_hand_speed_history = []
_sprint_buffer = 0
_was_sprinting_last_frame = False

# Overlay
_overlay_msg = ""
_overlay_timer = 0.0

# Head Tracking Calibration (disabled)
# calib_count = 0
# nose_x_samples = []
# nose_y_samples = []
# shoulder_mid_x_samples = []
# shoulder_mid_y_samples = []
# nose_minus_eye_y_samples = []

# Head Tracking Runtime (disabled)
# base_nose_x = 0.0
# base_nose_y = 0.0
# base_sh_x = 0.0
# base_sh_y = 0.0
# base_nose_minus_eye_y = 0.0
# smooth_head_x = 0.0
# smooth_head_y = 0.0

# Threading for Head (disabled)
# _desired_head_action = None
# _head_thread_stop = threading.Event()

# def _head_action_worker():
#     last_action = None
#     while not _head_thread_stop.is_set():
#         current_action = _desired_head_action
#         if current_action != last_action:
#             if current_action == "Turn Right":
#                 mouse_control.start_left() # Note: Camera inverted for intuition usually
#             elif current_action == "Turn Left":
#                 mouse_control.start_right()
#             else:
#                 mouse_control.stop_turning()
#             last_action = current_action
#         time.sleep(0.01)
#     mouse_control.stop_turning()

# _head_thread = threading.Thread(target=_head_action_worker, daemon=True)
# _head_thread.start()

# --- MEDIAPIPE SETUP ---
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

print("SYSTEM READY.")
print("  - RUNNING: Hip Level Stop Line + Tight Start Window.")
print("  - SPRINTING: Lift knees high while running.")
# print("  - HEAD TRACKING: Active.")

# =========================== MAIN LOOP ===========================
with PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w_px = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        # Detect
        pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Init Sprint Vars
        left_leg_lift = 1.0
        right_leg_lift = 1.0
        
        now = time.time()
        
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks[0]
            
            # Key Landmarks
            ls, rs = lm[11], lm[12] # Shoulders
            lw, rw = lm[15], lm[16] # Wrists
            lh, rh = lm[23], lm[24] # Hips
            left_eye, right_eye = lm[5], lm[2]
            
            # --- 1. HEAD TRACKING LOGIC (DISABLED) ---
            # nose = lm[0]
            # shoulder_mid_x = (ls.x + rs.x) / 2.0
            # shoulder_mid_y = (ls.y + rs.y) / 2.0
            # eye_mid_y = (left_eye.y + right_eye.y) / 2.0
            # 
            # head_action = "Forward"
            #
            # if calib_count < CONFIG["calib_frames_needed"]:
            #     nose_x_samples.append(nose.x)
            #     nose_y_samples.append(nose.y)
            #     shoulder_mid_x_samples.append(shoulder_mid_x)
            #     shoulder_mid_y_samples.append(shoulder_mid_y)
            #     nose_minus_eye_y_samples.append(nose.y - eye_mid_y)
            #     calib_count += 1
            #     head_action = "Calibrating..."
            # else:
            #     if calib_count == CONFIG["calib_frames_needed"]:
            #         base_nose_x = float(np.mean(nose_x_samples))
            #         base_nose_y = float(np.mean(nose_y_samples))
            #         base_sh_x = float(np.mean(shoulder_mid_x_samples))
            #         base_sh_y = float(np.mean(shoulder_mid_y_samples))
            #         base_nose_minus_eye_y = float(np.mean(nose_minus_eye_y_samples))
            #         calib_count += 1
            #
            #     rel_x = (nose.x - shoulder_mid_x) - (base_nose_x - base_sh_x)
            #     
            #     # Smooth
            #     alpha = CONFIG["head_smooth_alpha"]
            #     smooth_head_x = alpha * smooth_head_x + (1 - alpha) * rel_x
            #
            #     if smooth_head_x > CONFIG["head_turn_thresh_x"]: head_action = "Turn Left"
            #     elif smooth_head_x < -CONFIG["head_turn_thresh_x"]: head_action = "Turn Right"
            # 
            # _desired_head_action = head_action

            # --- 2. ZONES & VISUALS (RESTORED) ---
            start_thresh_y = ((ls.y + rs.y) / 2.0) + CONFIG["run_start_offset"]
            stop_thresh_y = ((lh.y + rh.y) / 2.0) + CONFIG["run_stop_offset"]
            click_y_limit = ((lm[5].y + lm[2].y)/2.0) - CONFIG["click_trigger_offset"]

            # Draw Lines
            y_start = int(start_thresh_y * h)
            y_stop = int(stop_thresh_y * h)
            cv2.line(frame, (0, y_start), (w_px, y_start), (255, 200, 0), 1) # Blue Start
            cv2.line(frame, (0, y_stop), (w_px, y_stop), (0, 0, 255), 2)     # Red Stop
            cv2.putText(frame, "STOP LINE", (10, y_stop - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw Click Zones (Blue Boxes)
            v_tol_px = int(CONFIG["click_vertical_align"] * w_px)
            ls_px = int(ls.x * w_px); rs_px = int(rs.x * w_px)
            cv2.rectangle(frame, (ls_px - v_tol_px, 0), (ls_px + v_tol_px, int(click_y_limit*h)), (50, 50, 255), 1)
            cv2.rectangle(frame, (rs_px - v_tol_px, 0), (rs_px + v_tol_px, int(click_y_limit*h)), (50, 50, 255), 1)

            # Draw Scroll Zones (Pink Boxes)
            h_tol_px = int(CONFIG["scroll_horizontal_align"] * h)
            ls_y_px = int(ls.y * h); rs_y_px = int(rs.y * h)
            cv2.rectangle(frame, (0, ls_y_px - h_tol_px), (ls_px - 100, ls_y_px + h_tol_px), (255, 0, 255), 1)
            cv2.rectangle(frame, (rs_px + 100, rs_y_px - h_tol_px), (w_px, rs_y_px + h_tol_px), (255, 0, 255), 1)

            # --- 3. GEOMETRY CHECKS (STRICT) ---
            # Vertical Alignment (For Clicking)
            l_is_vertical = abs(lw.x - ls.x) < CONFIG["click_vertical_align"]
            r_is_vertical = abs(rw.x - rs.x) < CONFIG["click_vertical_align"]
            l_click_active = l_is_vertical and (lw.y < click_y_limit)
            r_click_active = r_is_vertical and (rw.y < click_y_limit)

            # Horizontal Alignment (For Scrolling/Inventory)
            l_is_horizontal = abs(lw.y - ls.y) < CONFIG["scroll_horizontal_align"]
            r_is_horizontal = abs(rw.y - rs.y) < CONFIG["scroll_horizontal_align"]
            l_is_extended = abs(lw.x - ls.x) > CONFIG["tpose_extension"]
            r_is_extended = abs(rw.x - rs.x) > CONFIG["tpose_extension"]

            is_left_scroll = l_is_horizontal and l_is_extended and not l_click_active
            is_right_scroll = r_is_horizontal and r_is_extended and not r_click_active

            # --- 4. SCROLL & INVENTORY ---
            new_tpose = is_left_scroll and is_right_scroll

            if new_tpose:
                if not prev_tpose:
                    _safe_set_key(CONFIG["inventory_key"], True, now)
                    time.sleep(0.05)
                    _safe_set_key(CONFIG["inventory_key"], False, now)
                    _overlay_msg = "INVENTORY"
                    _overlay_timer = now + 1.5
                    # Toggle inventory state and gate
                    _inventory_open = not _inventory_open
                    _set_phone_gate(_inventory_open)
                prev_tpose = True
            elif is_left_scroll:
                prev_tpose = False
                if now - _last_scroll_time > CONFIG["scroll_cooldown_s"]:
                    force_scroll(-1) 
                    _last_scroll_time = now
                    _overlay_msg = "SCROLL RIGHT"
                    _overlay_timer = now + 0.5
            elif is_right_scroll:
                prev_tpose = False
                if now - _last_scroll_time > CONFIG["scroll_cooldown_s"]:
                    force_scroll(1)
                    _last_scroll_time = now
                    _overlay_msg = "SCROLL LEFT"
                    _overlay_timer = now + 0.5
            else:
                prev_tpose = False

            # --- 5. RUNNING LOGIC (STRICT) ---
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
                                _left_switch_count = 0; _right_switch_count = 0 

            # --- 6. JUMP LOGIC ---
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

            # --- 7. SPRINTING LOGIC ---
            current_lw_pos = (lw.x, lw.y)
            current_rw_pos = (rw.x, rw.y)
            
            hand_speed = 0.0
            # global _last_lw_pos, _last_rw_pos # Ensure these are accessible
            if _last_lw_pos and _last_rw_pos:
                l_dist = math.dist(current_lw_pos, _last_lw_pos)
                r_dist = math.dist(current_rw_pos, _last_rw_pos)
                hand_speed = (l_dist + r_dist) / 2.0
            
            _last_lw_pos = current_lw_pos
            _last_rw_pos = current_rw_pos

            _hand_speed_history.append(hand_speed)
            if len(_hand_speed_history) > 10:
                _hand_speed_history.pop(0)
            avg_speed = sum(_hand_speed_history) / len(_hand_speed_history)

            if avg_speed > CONFIG["sprint_speed_threshold"]:
                _is_sprinting = True
            elif avg_speed < CONFIG["walk_speed_threshold"]:
                _is_sprinting = False

            # --- 8. KEYBOARD EXECUTION (RUN/SPRINT) ---
            # If running logic says GO, check Sprinting logic
            if _running_holding:
                if _is_sprinting:
                    if not _was_sprinting_last_frame:
                        # Perform W Double Tap for Minecraft Sprint
                        key_up(CONFIG["running_key"])    
                        time.sleep(0.02) # Short blocking delay ok here
                        tap(CONFIG["running_key"], 0.05) 
                        time.sleep(0.02)
                        key_down(CONFIG["running_key"])  
                        _was_sprinting_last_frame = True
                    else:
                        set_key(CONFIG["running_key"], True)
                else:
                    # Normal walking
                    if _was_sprinting_last_frame:
                        # Reset to normal walk
                        key_up(CONFIG["running_key"])
                        time.sleep(0.05)
                        key_down(CONFIG["running_key"])
                        _was_sprinting_last_frame = False
                    else:
                        set_key(CONFIG["running_key"], True)
            else:
                set_key(CONFIG["running_key"], False)
                _was_sprinting_last_frame = False

            # --- 9. CLICK EXECUTION ---
            hold_time = CONFIG["hold_threshold_s"]
            # Left Hand
            if l_click_active:
                if not _prev_left_raised: _left_raise_start_time = now
                if (now - _left_raise_start_time) > hold_time and not _left_is_dragging:
                    force_down('left'); _left_is_dragging = True; _left_feedback_text = "DRAGGING"
            else:
                if _prev_left_raised:
                    if _left_is_dragging: force_up('left'); _left_is_dragging = False; _left_feedback_text = "RELEASED"
                    else: force_click('left'); _left_feedback_text = "CLICK"; _left_feedback_timer = now + 1.0
            _prev_left_raised = l_click_active

            # Right Hand
            if r_click_active:
                if not _prev_right_raised: _right_raise_start_time = now
                if (now - _right_raise_start_time) > hold_time and not _right_is_dragging:
                    force_down('right'); _right_is_dragging = True; _right_feedback_text = "DRAGGING"
            else:
                if _prev_right_raised:
                    if _right_is_dragging: force_up('right'); _right_is_dragging = False; _right_feedback_text = "RELEASED"
                    else: force_click('right'); _right_feedback_text = "CLICK"; _right_feedback_timer = now + 1.0
            _prev_right_raised = r_click_active

            # --- 10. UI OVERLAYS ---
            # Head status (disabled)
            # cv2.putText(frame, f"Head: {head_action}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Running/Sprinting Status
            status_text = "SPRINTING" if (_running_holding and _is_sprinting) else ("RUNNING" if _running_holding else "STOPPED")
            col = (0, 255, 255) if (_running_holding and _is_sprinting) else ((0, 255, 0) if _running_holding else (0, 0, 255))
            cv2.putText(frame, f"MODE: {status_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

            # --- NEW SPRINT SPEED METER ---
            target_speed = CONFIG["sprint_speed_threshold"]
            
            # Draw background box (Dark Grey)
            cv2.rectangle(frame, (w_px - 40, 300), (w_px - 20, 500), (50, 50, 50), -1)
            
            # Calculate fill based on hand speed (Multiplied by 1.5 so the bar feels full when you hit max speed)
            speed_fill = np.clip(avg_speed / (target_speed * 1.5), 0, 1)
            
            # Color: Cyan if sprinting, Green if just moving
            bar_color = (0, 255, 255) if _is_sprinting else (0, 255, 0)
            
            # Draw the actual bar
            cv2.rectangle(frame, (w_px - 40, 500), (w_px - 20, 500 - int(200 * speed_fill)), bar_color, -1)
            cv2.putText(frame, "SPEED", (w_px - 60, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Click Feedback
            if now < _left_feedback_timer: cv2.putText(frame, _left_feedback_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if now < _right_feedback_timer: cv2.putText(frame, _right_feedback_text, (w_px-200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Main Message Overlay
            if now < _overlay_timer:
                cv2.putText(frame, _overlay_msg, (int(w_px/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

            # Click Progress Bars
            if l_click_active and not _left_is_dragging:
                pct = min(1.0, (now - _left_raise_start_time)/hold_time)
                cv2.rectangle(frame, (50, 200), (50+int(100*pct), 210), (0,255,255), -1)
            if r_click_active and not _right_is_dragging:
                pct = min(1.0, (now - _right_raise_start_time)/hold_time)
                cv2.rectangle(frame, (w_px-150, 200), (w_px-150+int(100*pct), 210), (0,255,255), -1)

        cv2.imshow('Action Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

# Cleanup
# _head_thread_stop.set()
# if _head_thread is not None:
#     try: _head_thread.join(timeout=1.0)
#     except: pass
# try:
#     mouse_control.shutdown()
# except Exception:
#     pass
cap.release()
release_all()
cv2.destroyAllWindows()