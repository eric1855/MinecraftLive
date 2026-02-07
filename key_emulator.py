"""
Key emulator for sending WASD and basic game controls.
Uses key down/up so games (e.g. Minecraft) receive holdable movement keys.
On macOS: grant Terminal/Cursor Accessibility permission in
System Settings → Privacy & Security → Accessibility.
"""

import pyautogui

# Disable pyautogui fail-safe (moving mouse to corner to abort) so it doesn't interfere
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02  # small delay between key actions for stability

# Minecraft / FPS-style key names (pyautogui uses single-char or Key.xxx for special keys)
MINECRAFT_KEYS = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d", 
    "jump": "space",
    "sneak": "shift",
    "sprint": "ctrl",
    "inventory": "e",
    "drop": "q",
    "swap_hand": "f",
    # "attack" = left click (mouse), not keyboard
}

# Track which keys we've sent key_down for, so we only send key_up when released
_held: set[str] = set()


def key_down(key: str) -> None:
    """Hold a key down. Safe to call repeatedly for the same key."""
    k = key.lower().strip()
    if not k or k in _held:
        return
    try:
        pyautogui.keyDown(k)
        _held.add(k)
    except Exception as e:
        print(f"[key_emulator] keyDown({k!r}) failed: {e}")


def key_up(key: str) -> None:
    """Release a key. Safe to call even if key wasn't held."""
    k = key.lower().strip()
    if not k:
        return
    if k in _held:
        try:
            pyautogui.keyUp(k)
        except Exception as e:
            print(f"[key_emulator] keyUp({k!r}) failed: {e}")
        _held.discard(k)


def tap(key: str, duration: float = 0.02) -> None:
    """Press and release a key (for one-off actions like jump)."""
    key_down(key)
    import time
    time.sleep(duration)
    key_up(key)


def set_key(key: str, pressed: bool) -> None:
    """Set key state: pressed=True -> key_down, pressed=False -> key_up.
    Use this to drive movement from pose/gesture (only sends events on change)."""
    if pressed:
        key_down(key)
    else:
        key_up(key)


def release_all() -> None:
    """Release every key we think we're holding. Call on exit or when stopping control."""
    for k in list(_held):
        key_up(k)


def get_held() -> set[str]:
    """Return set of keys currently held (for debugging)."""
    return set(_held)


# Convenience: movement as a group
def set_movement(forward: bool = False, back: bool = False, left: bool = False, right: bool = False) -> None:
    """Set W/A/S/D state from booleans. Only one of forward/back and one of left/right makes sense."""
    set_key("w", forward)
    set_key("s", back)
    set_key("a", left)
    set_key("d", right)


if __name__ == "__main__":
    import time
    print("Key emulator: 3-second delay, then W hold 2s, then release all.")
    print("Focus a text field or Minecraft window.")
    time.sleep(3)
    key_down("w")
    time.sleep(2)
    release_all()
    print("Done.")