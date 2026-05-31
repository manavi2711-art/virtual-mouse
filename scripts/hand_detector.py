"""
Hand Detector Module
====================
Uses MediaPipe Hands to detect and track hand landmarks in real-time.
Provides 21 landmark points per hand for gesture recognition.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Wraps MediaPipe Hands for easy hand detection and landmark extraction.
    """

    # Finger tip landmark IDs
    TIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def __init__(self, max_hands=1, detection_conf=0.8, tracking_conf=0.8):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=2, circle_radius=2)
        self.results = None

    def find_hands(self, frame, draw=True):
        """Detect hands in a BGR frame and optionally draw landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if self.results.multi_hand_landmarks and draw:
            for hand_lm in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.draw_spec,
                    self.draw_spec
                )

        return frame

    def get_landmarks(self, frame):
        """
        Returns list of (x, y) pixel coordinates for all 21 landmarks.
        Returns empty list if no hand detected.
        """
        landmarks = []

        if self.results and self.results.multi_hand_landmarks:
            h, w, _ = frame.shape

            hand = self.results.multi_hand_landmarks[0]

            for lm in hand.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))

        return landmarks

    def get_normalized_landmarks(self):
        """
        Returns flattened normalized [x, y, z] coordinates (63 values).
        Used as feature vector for the MLP gesture classifier.
        """

        if self.results and self.results.multi_hand_landmarks:

            hand = self.results.multi_hand_landmarks[0]

            coords = []

            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            return np.array(coords, dtype=np.float32)

        return None

    def fingers_up(self, landmarks):
        """
        Returns a list of 5 booleans:
        [Thumb, Index, Middle, Ring, Pinky]

        1 = finger up
        0 = finger down
        """

        if len(landmarks) < 21:
            return []

        fingers = []

        # Thumb
        if landmarks[self.TIP_IDS[0]][0] < landmarks[self.TIP_IDS[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for tip_id in self.TIP_IDS[1:]:

            if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def hand_detected(self):
        """Returns True if at least one hand is detected."""

        return bool(self.results and self.results.multi_hand_landmarks)


# TESTING MODULE
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        detector.find_hands(frame)

        landmarks = detector.get_landmarks(frame)

        if landmarks:
            print("Index Finger Tip:", landmarks[8])

        cv2.imshow("Hand Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()