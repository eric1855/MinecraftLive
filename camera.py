import cv2
Interpreter = None
try:
    # Prefer the lightweight tflite-runtime if available
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        # Common TensorFlow location
        from tensorflow.lite import Interpreter
    except Exception:
        try:
            # Alternate TensorFlow path
            from tensorflow.lite.python.interpreter import Interpreter
        except Exception:
            Interpreter = None
import numpy as np
import os
import threading
import time
from collections import deque

from key_emulator import set_key, release_all

# ============== TWEAKABLE CONFIG (adjust for your camera / pose) ==============
CONFIG = {
    # --- Running = hands moving in opposite directions for t > min time ---
    # Landmark indices: 15=left wrist, 16=right wrist (MediaPipe pose)
    # MoveNet (COCO) keypoint indices: 9=left_wrist, 10=right_wrist
    "hand_left_wrist": 9,
    "hand_right_wrist": 10,
    # Frames of history used to compute "is this hand moving up or down?" (y in image)
    "running_velocity_window_frames": 3,
    # Minimum vertical motion (normalized y per frame) to count as "moving"
    # Increase if noise triggers; decrease if real pumps are ignored
    "running_min_velocity": 0.0015,
    # Running = opposite motion for at least this many consecutive frames (t > x)
    # Higher = must pump longer before W engages; lower = quicker response
    "running_opposite_direction_min_frames": 2,

    # --- Smoothing (reduces jitter after we've already decided "running") ---
    "running_confirm_frames": 1,
    "running_release_frames": 12,
    "running_key": "w",
    # Downscale factor for inference to speed up model (0.5 = half size). 1.0 = no downscale.
    "inference_downscale": 0.6,
    # Maximum inference FPS (set to 0 for unlimited)
    "max_inference_fps": 30,
    # MoveNet model path (place movenet_thunder.tflite in the workspace)
    "movenet_model": "movenet_thunder.tflite",
}
# =============================================================================

def _load_tflite_interpreter(model_path: str):
    if Interpreter is None:
        raise RuntimeError("No TFLite interpreter available. Install tflite-runtime or tensorflow.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MoveNet model not found: {model_path}. Download and place it in the workspace.")
    interp = Interpreter(model_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    # Expecting input shape [1, H, W, 3]
    _, in_h, in_w, _ = input_details[0]["shape"]
    return interp, (in_w, in_h)

cap = cv2.VideoCapture(0)

# If the camera failed to initialize (common on macOS when camera permission
# hasn't been granted), try a few fallbacks and print actionable guidance.
if not cap.isOpened():
    print("OpenCV: camera failed to initialize. Trying AVFoundation backend...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    except Exception:
        pass

if not cap.isOpened():
    # Try a sample video file in the workspace as a fallback for testing
    for fname in ("sample_video.mp4", "test_video.mp4", "sample.mp4"):
        if os.path.exists(fname):
            print(f"Opening fallback video file {fname}")
            cap = cv2.VideoCapture(fname)
            break

if not cap.isOpened():
    print("Camera not available. On macOS grant Terminal camera access:")
    print("System Settings → Privacy & Security → Camera → enable Terminal or your shell app")
    print("Or reset camera permissions and re-run to trigger a prompt:")
    print("  tccutil reset Camera")
    raise SystemExit(1)

frame_count = 0

# Shared buffer for latest frame (capture thread writes, main thread reads)
_latest_frame = None
_frame_lock = threading.Lock()
_stop_event = threading.Event()


def _capture_loop():
    global _latest_frame
    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with _frame_lock:
            _latest_frame = frame


# Smoothing buffer: True = running detected this frame (start with not-running)
_n = max(CONFIG["running_confirm_frames"], CONFIG["running_release_frames"])
_running_buffer = deque([False] * _n, maxlen=_n)

# Hand motion: recent y positions for velocity (smaller y = higher in image)
_w = CONFIG["running_velocity_window_frames"] + 1
_left_hand_ys: deque = deque(maxlen=_w)
_right_hand_ys: deque = deque(maxlen=_w)
_opposite_motion_streak: int = 0

# Connections for MoveNet (COCO 17-keypoint layout)
POSE_CONNECTIONS = [
    (5,7),(7,9),  # left arm
    (6,8),(8,10), # right arm
    (5,6),        # shoulders
    (11,12),      # hips
    (5,11),(6,12),# side torso
    (11,13),(13,15), # left leg
    (12,14),(14,16), # right leg
]


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


def _run_movenet_loop():
    interp, (in_w, in_h) = _load_tflite_interpreter(CONFIG["movenet_model"])
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Start capture thread to always keep the latest camera frame (reduces latency)
    cap_thread = threading.Thread(target=_capture_loop, daemon=True)
    cap_thread.start()
    last_inference_time = 0.0
    try:
        while True:
            with _frame_lock:
                frame = None if _latest_frame is None else _latest_frame.copy()
            if frame is None:
                time.sleep(0.005)
                continue

            max_fps = CONFIG.get("max_inference_fps", 0)
            now = time.time()
            if max_fps and (now - last_inference_time) < (1.0 / max_fps):
                continue
            last_inference_time = now

            frame_count_local = None
            frame_count_local = 0
            frame_count_local += 1

            h, w = frame.shape[:2]

            # Resize to model input size (MoveNet expects square input)
            proc = cv2.resize(frame, (in_w, in_h))
            inp = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            inp = inp.astype(np.float32)
            # Normalize to [0,1]
            inp = inp / 255.0
            inp = np.expand_dims(inp, axis=0)

            # Set input tensor
            interp.set_tensor(input_details[0]['index'], inp)
            interp.invoke()
            out = interp.get_tensor(output_details[0]['index'])

            # MoveNet outputs either (1,1,17,3) or (1,17,3)
            kps = None
            if out.ndim == 4:
                # (1,1,17,3)
                kps = out[0,0,:,:]
            elif out.ndim == 3:
                # (1,17,3)
                kps = out[0,:,:]
            else:
                raise RuntimeError(f"Unexpected MoveNet output shape: {out.shape}")

            # Build simple landmark objects with x,y,score (normalized)
            class _KP:
                def __init__(self, x, y, score):
                    self.x = x
                    self.y = y
                    self.score = score

            landmarks = []
            for kp in kps:
                y, x, score = float(kp[0]), float(kp[1]), float(kp[2])
                # MoveNet returns y,x,score
                landmarks.append(_KP(x, y, score))

            # Draw skeleton and keypoints on original frame
            for connection in POSE_CONNECTIONS:
                a = landmarks[connection[0]]
                b = landmarks[connection[1]]
                cv2.line(frame, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)), (0, 255, 0), 2)
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)

            # Simple gesture: wrists above shoulders -> "Gangnam Style"
            left_wrist_idx = CONFIG["hand_left_wrist"]
            right_wrist_idx = CONFIG["hand_right_wrist"]
            left_shoulder_idx = 5
            right_shoulder_idx = 6
            action = "None"
            running_raw = False
            if landmarks[left_wrist_idx].y < landmarks[left_shoulder_idx].y and landmarks[right_wrist_idx].y < landmarks[right_shoulder_idx].y:
                action = "Gangnam Style"
            else:
                running_raw = _detect_running_raw(landmarks)
                if running_raw:
                    action = "Running in Place (W held)"

            running_smoothed = _running_after_smoothing(running_raw)
            set_key(CONFIG["running_key"], running_smoothed)

            cv2.putText(frame, action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Action Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        _stop_event.set()
        cap_thread.join(timeout=1)


if __name__ == '__main__':
    _run_movenet_loop()

cap.release()
release_all()
cv2.destroyAllWindows()
