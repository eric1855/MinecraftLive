import subprocess
import os
import signal

# Track the currently running AppleScript process
_current_process = None

def _start_loop(key_code):
    """
    Starts a background AppleScript process that batches key presses
    to maximize speed.
    """
    global _current_process
    stop_turning()
    
    # We repeat the 'key code' command multiple times inside the loop.
    # This reduces the "loop overhead" significantly.
    # We also REMOVED the 'delay' command entirely.
    script = f"""
    osascript -e '
        tell application "System Events"
            repeat 
                delay(0.001)
                key code {key_code}
                key code {key_code}
            end repeat
        end tell
    '
    """
    
    _current_process = subprocess.Popen(
        script, 
        shell=True, 
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )

def start_left():
    # 86 is Numpad 4 (Left) for Mouse Keys
    _start_loop(86)

def start_right():
    # 88 is Numpad 6 (Right) for Mouse Keys
    _start_loop(88)

def stop_turning():
    global _current_process
    if _current_process:
        try:
            # Send SIGTERM to the process group to kill the AppleScript loop instantly
            os.killpg(os.getpgid(_current_process.pid), signal.SIGTERM)
        except Exception:
            pass
        _current_process = None

# Compatibility wrapper (optional, does nothing now as we use start/stop)
def send_key_code_once(key):
    pass