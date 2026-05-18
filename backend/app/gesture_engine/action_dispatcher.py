"""Map gesture labels to OS actions with debouncing."""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAPPINGS: dict[str, str] = {
    "thumbs_up": "volume_up",
    "thumbs_down": "volume_down",
    "open_palm": "play_pause",
    "closed_fist": "mute_toggle",
    "peace_sign": "screenshot",
    "pinch": "zoom_in",
    "spread": "zoom_out",
    "swipe_left": "previous",
    "swipe_right": "next",
    "swipe_up": "scroll_up",
    "wave": "play_pause",
}

# Human-readable for API
ACTION_DESCRIPTIONS: dict[str, str] = {
    "volume_up": "Volume up (+5%)",
    "volume_down": "Volume down (-5%)",
    "play_pause": "Play / pause media",
    "mute_toggle": "Mute toggle",
    "screenshot": "Screenshot",
    "zoom_in": "Zoom in",
    "zoom_out": "Zoom out",
    "previous": "Previous slide / track",
    "next": "Next slide / track",
    "scroll_up": "Scroll up",
    "none": "No action",
}


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "mappings.json"


class ActionDispatcher:
    def __init__(
        self,
        gesture_to_action: dict[str, str] | None = None,
        debounce_ms: float = 500.0,
        dry_run: bool = False,
    ) -> None:
        self.gesture_to_action = dict(DEFAULT_MAPPINGS)
        if gesture_to_action:
            self.gesture_to_action.update(gesture_to_action)
        self.debounce_ms = debounce_ms
        self.dry_run = dry_run
        self._last_fire: dict[str, float] = {}
        self._load_overrides()

    def _load_overrides(self) -> None:
        p = _config_path()
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.gesture_to_action.update({k: str(v) for k, v in data.items()})
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load mappings.json: %s", e)

    def save_overrides(self) -> None:
        p = _config_path()
        p.write_text(json.dumps(self.gesture_to_action, indent=2), encoding="utf-8")

    def update_mapping(self, gesture: str, action: str) -> None:
        self.gesture_to_action[gesture] = action
        self.save_overrides()

    def get_mappings(self) -> dict[str, str]:
        return dict(self.gesture_to_action)

    def action_for_gesture(self, gesture: str) -> str | None:
        if gesture == "UNKNOWN":
            return None
        return self.gesture_to_action.get(gesture)

    def dispatch(self, gesture: str, confidence: float) -> str | None:
        if gesture == "UNKNOWN":
            return None
        action = self.action_for_gesture(gesture)
        if action is None:
            return None
        now = time.monotonic() * 1000.0
        last = self._last_fire.get(gesture, 0.0)
        if now - last < self.debounce_ms:
            return None
        self._last_fire[gesture] = now
        if self.dry_run:
            logger.info("[dry-run] dispatch %s -> %s (conf=%.2f)", gesture, action, confidence)
            return action
        _execute_action(action)
        return action


def _execute_action(action: str) -> None:
    try:
        import pyautogui  # noqa: PLC0415

        pyautogui.FAILSAFE = False
    except ImportError:
        logger.warning("pyautogui not available")
        return

    system = platform.system()

    if action == "volume_up":
        pyautogui.press("volumeup")
    elif action == "volume_down":
        pyautogui.press("volumedown")
    elif action == "play_pause":
        pyautogui.press("playpause")
    elif action == "mute_toggle":
        if system == "Darwin":
            pyautogui.hotkey("command", "shift", "m")
        else:
            pyautogui.hotkey("ctrl", "m")
    elif action == "screenshot":
        if system == "Darwin":
            pyautogui.hotkey("command", "shift", "3")
        else:
            pyautogui.hotkey("ctrl", "shift", "s")
    elif action == "zoom_in":
        if system == "Darwin":
            pyautogui.hotkey("command", "+")
        else:
            pyautogui.hotkey("ctrl", "+")
    elif action == "zoom_out":
        if system == "Darwin":
            pyautogui.hotkey("command", "-")
        else:
            pyautogui.hotkey("ctrl", "-")
    elif action == "previous":
        pyautogui.press("left")
    elif action == "next":
        pyautogui.press("right")
    elif action == "scroll_up":
        pyautogui.scroll(3)
    else:
        logger.debug("No executor for action %s", action)
