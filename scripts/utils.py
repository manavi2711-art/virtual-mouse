"""
Utility Functions
=================
Helper classes and functions for the Virtual Mouse system.
Includes cursor smoothing, distance calculation, and FPS tracking.
"""

import numpy as np
import time
import cv2


class Smoother:
    """
    Exponential moving average smoother to reduce cursor jitter.
    Higher alpha = more responsive but jittery.
    Lower alpha = smoother but laggy.
    """

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def smooth(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        sx = int(self.alpha * x + (1 - self.alpha) * self.prev_x)
        sy = int(self.alpha * y + (1 - self.alpha) * self.prev_y)
        self.prev_x, self.prev_y = sx, sy
        return sx, sy

    def reset(self):
        self.prev_x = None
        self.prev_y = None


class MovingAverageSmoother:
    """
    Buffer-based moving average smoother. Smoother but higher latency.
    """

    def __init__(self, buffer_size=5):
        self.buffer = []
        self.size = buffer_size

    def smooth(self, point):
        self.buffer.append(point)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        return (
            int(np.mean([p[0] for p in self.buffer])),
            int(np.mean([p[1] for p in self.buffer]))
        )


class FPSCounter:
    """Tracks and displays frames per second."""

    def __init__(self, avg_frames=30):
        self.times = []
        self.avg_frames = avg_frames

    def tick(self):
        self.times.append(time.time())
        if len(self.times) > self.avg_frames:
            self.times.pop(0)

    def get_fps(self):
        if len(self.times) < 2:
            return 0
        elapsed = self.times[-1] - self.times[0]
        return int((len(self.times) - 1) / elapsed) if elapsed > 0 else 0

    def draw(self, frame, x=10, y=30):
        fps = self.get_fps()
        color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def euclidean_distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_hud(frame, action, fingers, gesture_label=None):
    """Draw a semi-transparent HUD overlay on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # HUD background
    cv2.rectangle(overlay, (0, h - 110), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Action text
    cv2.putText(frame, f"Action: {action}", (10, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)

    # Finger states
    finger_names = ["T", "I", "M", "R", "P"]
    for i, (name, state) in enumerate(zip(finger_names, fingers if fingers else [0]*5)):
        color = (0, 255, 0) if state else (80, 80, 80)
        cv2.putText(frame, name, (10 + i * 35, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Gesture label from classifier
    if gesture_label:
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

    # Controls hint
    cv2.putText(frame, "Q: Quit | P: Pause | T: Train Mode", (w - 320, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    return frame