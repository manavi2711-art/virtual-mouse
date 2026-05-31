import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = "/tmp/hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

TIP_IDS = [4, 8, 12, 16, 20]

class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        _ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=tracking_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.results = self.landmarker.detect(mp_image)

        if draw and self.results.hand_landmarks:
            h, w = frame.shape[:2]
            for hand in self.results.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                connections = vision.HandLandmarker.HAND_CONNECTIONS if hasattr(vision.HandLandmarker, 'HAND_CONNECTIONS') else mp.solutions.hands.HAND_CONNECTIONS
                for i, j in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                              (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                              (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]:
                    cv2.line(frame, pts[i], pts[j], (0, 200, 100), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        return frame

    def get_landmarks(self, frame):
        if not self.results or not self.results.hand_landmarks:
            return []
        h, w = frame.shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in self.results.hand_landmarks[0]]

    def get_normalized_landmarks(self):
        if not self.results or not self.results.hand_landmarks:
            return None
        coords = []
        for lm in self.results.hand_landmarks[0]:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)

    def fingers_up(self, landmarks):
        if len(landmarks) < 21:
            return []
        fingers = []
        fingers.append(1 if landmarks[TIP_IDS[0]][0] < landmarks[TIP_IDS[0]-1][0] else 0)
        for tip_id in TIP_IDS[1:]:
            fingers.append(1 if landmarks[tip_id][1] < landmarks[tip_id-2][1] else 0)
        return fingers

    def hand_detected(self):
        return bool(self.results and self.results.hand_landmarks)
