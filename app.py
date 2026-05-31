import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from hand_detector import HandDetector
from gesture_classifier import rule_based_predict

app = FastAPI()
detector = HandDetector(max_hands=1, detection_conf=0.7, tracking_conf=0.7)


class FrameRequest(BaseModel):
    image: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_frame(req: FrameRequest):
    img_data = base64.b64decode(req.image.split(",")[-1])
    frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"gesture": "none", "fingers": [], "landmarks": []}

    detector.find_hands(frame, draw=True)
    landmarks = detector.get_landmarks(frame)
    fingers = detector.fingers_up(landmarks)
    gesture = rule_based_predict(fingers, landmarks)

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    annotated = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

    return {"gesture": gesture, "fingers": fingers,
            "landmarks": [[x, y] for x, y in landmarks],
            "annotated_frame": annotated}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
