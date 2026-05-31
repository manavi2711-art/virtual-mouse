"""
Gesture Controller
==================
Maps hand landmarks and finger states to mouse actions.
Supports: Move, Left Click, Right Click, Double Click,
          Drag & Drop, Scroll Up/Down, Volume Control.
"""

import pyautogui
import time
import numpy as np
from utils import Smoother, euclidean_distance

# Disable PyAutoGUI fail-safe (move mouse to corner to stop)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove default pause between calls


class GestureController:
    """
    Core gesture-to-action mapper.

    Gesture Map:
    ┌─────────────────────────────────────┬──────────────────┐
    │ Finger State [T,I,M,R,P]            │ Action           │
    ├─────────────────────────────────────┼──────────────────┤
    │ [0,1,0,0,0]                         │ Move cursor      │
    │ [0,1,1,0,0] + fingers close         │ Left Click       │
    │ [1,1,0,0,0] + pinch                 │ Right Click      │
    │ [0,1,0,0,0] + rapid tap             │ Double Click     │
    │ [0,1,1,0,0] + hold close            │ Drag & Drop      │
    │ [0,0,0,0,1]                         │ Scroll Up        │
    │ [1,0,0,0,1]                         │ Scroll Down      │
    │ [1,1,1,1,1]                         │ Volume Control   │
    │ [0,0,0,0,0]                         │ Pause / Stop     │
    └─────────────────────────────────────┴──────────────────┘
    """

    CLICK_DIST_THRESHOLD = 38    # px — fingers close enough to click
    PINCH_DIST_THRESHOLD = 40    # px — thumb+index pinch threshold
    DRAG_HOLD_FRAMES = 15        # frames to hold before drag starts
    CLICK_COOLDOWN = 0.35        # seconds between clicks
    DOUBLE_CLICK_WINDOW = 0.4    # seconds for double click detection
    SCROLL_SPEED = 3             # scroll units per frame
    VOLUME_SENSITIVITY = 150     # pinch px range → volume 0-100

    def __init__(self, screen_w, screen_h, cam_w, cam_h,
                 margin=80, smoothing=0.35):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.margin = margin

        self.smoother = Smoother(alpha=smoothing)

        # State tracking
        self.prev_click_time = 0
        self.last_click_time = 0
        self.dragging = False
        self.drag_hold_count = 0
        self.prev_action = None
        self.paused = False

    def map_to_screen(self, x, y):
        """Map camera coordinates (with margin) → screen coordinates."""
        m = self.margin
        x = max(m, min(self.cam_w - m, x))
        y = max(m, min(self.cam_h - m, y))
        sx = int((x - m) / (self.cam_w - 2 * m) * self.screen_w)
        sy = int((y - m) / (self.cam_h - 2 * m) * self.screen_h)
        return sx, sy

    def process(self, landmarks, fingers):
        """
        Process current landmarks and finger states → execute mouse action.
        Returns action label string.
        """
        if self.paused:
            return "⏸ Paused"

        if not landmarks or len(landmarks) < 21:
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            return "No Hand"

        if not fingers:
            return "No Hand"

        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]

        now = time.time()

        # ── FIST: Pause/Stop ──────────────────────────────────────────
        if fingers == [0, 0, 0, 0, 0]:
            if self.dragging:
                pyautogui.mouseUp()
                self.dragging = False
            return "✊ Stop"

        # ── MOVE: Only index finger up ────────────────────────────────
        if fingers == [0, 1, 0, 0, 0]:
            sx, sy = self.map_to_screen(*index_tip)
            sx, sy = self.smoother.smooth(sx, sy)

            if self.dragging:
                pyautogui.dragTo(sx, sy, button='left', duration=0)
                return "🖱 Dragging"
            else:
                pyautogui.moveTo(sx, sy, duration=0)
                return "🖱 Move"

        # ── LEFT CLICK / DOUBLE CLICK / DRAG ─────────────────────────
        if fingers == [0, 1, 1, 0, 0]:
            d = euclidean_distance(index_tip, middle_tip)

            if d < self.CLICK_DIST_THRESHOLD:
                # Drag detection: hold close for N frames
                self.drag_hold_count += 1

                if self.drag_hold_count >= self.DRAG_HOLD_FRAMES:
                    if not self.dragging:
                        pyautogui.mouseDown(button='left')
                        self.dragging = True
                    return "🔗 Drag Hold"

                # Double click detection
                if now - self.last_click_time < self.DOUBLE_CLICK_WINDOW:
                    if now - self.prev_click_time > self.CLICK_COOLDOWN:
                        pyautogui.doubleClick()
                        self.prev_click_time = now
                        self.last_click_time = 0
                        self.drag_hold_count = 0
                        return "👆 Double Click"

                # Single left click
                if now - self.prev_click_time > self.CLICK_COOLDOWN:
                    pyautogui.click()
                    self.last_click_time = now
                    self.prev_click_time = now
                    self.drag_hold_count = 0
                    return "👆 Left Click"

                return "Ready to Click"
            else:
                # Fingers spread → release drag
                self.drag_hold_count = 0
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                return "🖱 Move (2-finger)"

        # ── RIGHT CLICK: Thumb + Index pinch ─────────────────────────
        if fingers == [1, 1, 0, 0, 0]:
            d = euclidean_distance(thumb_tip, index_tip)
            if d < self.PINCH_DIST_THRESHOLD:
                if now - self.prev_click_time > self.CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    self.prev_click_time = now
                    return "🖱 Right Click"
            return "Right Click Ready"

        # ── SCROLL UP: Pinky only ─────────────────────────────────────
        if fingers == [0, 0, 0, 0, 1]:
            pyautogui.scroll(self.SCROLL_SPEED)
            return "⬆ Scroll Up"

        # ── SCROLL DOWN: Thumb + Pinky ────────────────────────────────
        if fingers == [1, 0, 0, 0, 1]:
            pyautogui.scroll(-self.SCROLL_SPEED)
            return "⬇ Scroll Down"

        # ── VOLUME CONTROL: All fingers up ───────────────────────────
        if fingers == [1, 1, 1, 1, 1]:
            d = euclidean_distance(thumb_tip, pinky_tip)
            volume = int(np.clip(d / self.VOLUME_SENSITIVITY * 100, 0, 100))
            try:
                # Works on Linux with pactl
                import subprocess
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"],
                    capture_output=True
                )
            except Exception:
                pass
            return f"🔊 Volume: {volume}%"

        return "Idle"

    def toggle_pause(self):
        self.paused = not self.paused
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False