import threading
import time
from pynput.mouse import Controller

# Mouse controller
_mouse = Controller()

# Threaded smooth turning to avoid OS key spam
_turn_thread = None
_turn_stop = threading.Event()
_turn_lock = threading.Lock()
_turn_dir = 0  # -1 = left, +1 = right, 0 = idle

# Tuning
_step_px = 4
_sleep_s = 0.01


def _turn_worker():
    while not _turn_stop.is_set():
        with _turn_lock:
            direction = _turn_dir
        if direction != 0:
            _mouse.move(direction * _step_px, 0)
        time.sleep(_sleep_s)


def _ensure_thread():
    global _turn_thread
    if _turn_thread is None or not _turn_thread.is_alive():
        _turn_stop.clear()
        _turn_thread = threading.Thread(target=_turn_worker, daemon=True)
        _turn_thread.start()


def start_left():
    _ensure_thread()
    with _turn_lock:
        global _turn_dir
        _turn_dir = -1


def start_right():
    _ensure_thread()
    with _turn_lock:
        global _turn_dir
        _turn_dir = 1


def stop_turning():
    with _turn_lock:
        global _turn_dir
        _turn_dir = 0


def shutdown():
    _turn_stop.set()
    if _turn_thread is not None:
        try:
            _turn_thread.join(timeout=0.5)
        except Exception:
            pass


# Compatibility wrapper (optional)
def send_key_code_once(key):
    pass