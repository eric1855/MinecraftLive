#!/usr/bin/env python3
"""
Test the key emulator: drive WASD + space with number keys 8/4/5/6 and 0.
- 8=W, 5=S, 4=A, 6=D, 0=Space (jump)
- Press 'r' to release all and quit.
Run this, then focus Minecraft or a text editor to see keys being sent.
"""

import sys
import time

try:
    import pyautogui
except ImportError:
    print("Install pyautogui: pip install pyautogui")
    sys.exit(1)

# Use our emulator
from key_emulator import (
    set_key,
    set_movement,
    tap,
    release_all,
    get_held,
)

def main():
    try:
        from pynput import keyboard
    except ImportError:
        print("For interactive test, install pynput: pip install pynput")
        print("Falling back to auto demo: W 2s, then A 2s, then release.")
        time.sleep(2)
        set_key("w", True)
        time.sleep(2)
        set_key("w", False)
        set_key("a", True)
        time.sleep(2)
        release_all()
        print("Demo done. Held was:", get_held())
        return

    print("Key emulator test. Focus Minecraft or a text window.")
    print("  Num 8=W, 5=S, 4=A, 6=D, 0=Space, R=release all & quit")
    print("  (Use numpad or main keys 8/5/4/6/0)")

    state = {"w": False, "a": False, "s": False, "d": False, "space": False}

    def on_press(k):
        try:
            key = k.char
        except AttributeError:
            key = getattr(k, "name", "")

        if key in ("8", "w"):
            state["w"] = True
            set_key("w", True)
        elif key in ("5", "s"):
            state["s"] = True
            set_key("s", True)
        elif key in ("4", "a"):
            state["a"] = True
            set_key("a", True)
        elif key in ("6", "d"):
            state["d"] = True
            set_key("d", True)
        elif key in ("0", " "):
            state["space"] = True
            set_key("space", True)
        elif key and key.lower() == "r":
            release_all()
            state.update(w=False, a=False, s=False, d=False, space=False)
            return False  # stop listener and exit

    def on_release(k):
        try:
            key = k.char
        except AttributeError:
            key = getattr(k, "name", "")
        if key in ("8", "w"):
            state["w"] = False
            set_key("w", False)
        elif key in ("5", "s"):
            state["s"] = False
            set_key("s", False)
        elif key in ("4", "a"):
            state["a"] = False
            set_key("a", False)
        elif key in ("6", "d"):
            state["d"] = False
            set_key("d", False)
        elif key in ("0", " "):
            state["space"] = False
            set_key("space", False)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    release_all()
    print("Quit. All keys released.")


if __name__ == "__main__":
    main()
