"""
Virtual Mouse Using Hand Gestures
===================================
Main entry point for the real-time virtual mouse application.

Usage:
    python main.py [--no-model] [--cam 0] [--width 640] [--height 480]

Options:
    --no-model    Use rule-based gestures only (no trained model needed)
    --cam INT     Camera index (default: 0)
    --width INT   Camera capture width
    --height INT  Camera capture height

Controls (while running):
    Q / ESC       Quit
    P             Pause / Resume
    D             Toggle debug overlay
    M             Toggle classifier mode (model vs rules)
"""

import cv2
import pyautogui
import argparse
import sys
import time
import numpy as np

from hand_detector import HandDetector
from gesture_controller import GestureController
from gesture_classifier import GestureClassifier, rule_based_predict
from utils import FPSCounter, draw_hud


def parse_args():
    p = argparse.ArgumentParser(description="Virtual Mouse Using Hand Gestures")
    p.add_argument("--no-model", action="store_true",
                   help="Skip loading ML model, use rule-based gestures")
    p.add_argument("--cam", type=int, default=0, help="Camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


def draw_debug(frame, landmarks, fingers, gesture_label, confidence):
    """Extra debug overlay — landmark indices, confidence bar."""
    if landmarks:
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
            if i in [4, 8, 12, 16, 20]:  # tips
                cv2.circle(frame, (x, y), 6, (0, 140, 255), 2)

    if confidence > 0:
        bar_w = int(confidence * 150)
        cv2.rectangle(frame, (frame.shape[1] - 165, 50),
                      (frame.shape[1] - 165 + bar_w, 65), (0, 200, 100), -1)
        cv2.rectangle(frame, (frame.shape[1] - 165, 50),
                      (frame.shape[1] - 15, 65), (100, 100, 100), 1)
        cv2.putText(frame, f"{confidence*100:.0f}%",
                    (frame.shape[1] - 160, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def main():
    args = parse_args()

    print("\n🖐  Virtual Mouse — Hand Gesture Control")
    print("=" * 45)
    print(f"Camera   : {args.cam}")
    print(f"Resolution: {args.width}×{args.height}")

    # Screen size
    SCREEN_W, SCREEN_H = pyautogui.size()
    print(f"Screen   : {SCREEN_W}×{SCREEN_H}")

    # Load ML model
    classifier = GestureClassifier()
    use_model = False

    if not args.no_model:
        print("\nLoading gesture classifier...")
        use_model = classifier.load()
        if not use_model:
            print("⚠️  Falling back to rule-based gestures.")
            print("   Run: python collect_data.py && python train.py")
    else:
        print("Using rule-based gesture recognition.")

    # Init camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.cam}")
        sys.exit(1)

    # Init modules
    detector = HandDetector(max_hands=1, detection_conf=0.8, tracking_conf=0.8)
    controller = GestureController(
        screen_w=SCREEN_W, screen_h=SCREEN_H,
        cam_w=args.width, cam_h=args.height
    )
    fps_counter = FPSCounter()

    debug_mode = False
    model_mode = use_model
    gesture_label = ""
    confidence = 0.0

    print("\n✅ System ready! Controls: Q=Quit | P=Pause | D=Debug | M=Toggle Model")
    print("=" * 45 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel

        # Detect hands
        frame = detector.find_hands(frame, draw=True)
        landmarks = detector.get_landmarks(frame)
        fingers = detector.fingers_up(landmarks)

        # Gesture recognition
        if model_mode and use_model:
            lm_vec = detector.get_normalized_landmarks()
            if lm_vec is not None:
                gesture_label, confidence = classifier.predict(lm_vec)
            else:
                gesture_label, confidence = "", 0.0
        else:
            gesture_label = rule_based_predict(fingers, landmarks)
            confidence = 0.0

        # Execute mouse action
        action = controller.process(landmarks, fingers)

        # HUD
        frame = draw_hud(frame, action, fingers, gesture_label)
        fps_counter.tick()
        fps_counter.draw(frame)

        # Debug overlay
        if debug_mode:
            frame = draw_debug(frame, landmarks, fingers, gesture_label, confidence)

        # Mode indicators
        mode_text = f"{'[MODEL]' if model_mode else '[RULES]'}"
        color = (0, 200, 255) if model_mode else (255, 160, 0)
        cv2.putText(frame, mode_text, (frame.shape[1] - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if controller.paused:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.putText(frame, "PAUSED — Press P to Resume", (w//2 - 180, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 3)

        cv2.imshow("Virtual Mouse | Hand Gesture Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('p') or key == ord('P'):
            controller.toggle_pause()
            print("⏸ Paused" if controller.paused else "▶ Resumed")
        elif key == ord('d') or key == ord('D'):
            debug_mode = not debug_mode
            print(f"🔍 Debug: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('m') or key == ord('M'):
            if use_model:
                model_mode = not model_mode
                print(f"🔄 Mode: {'Model' if model_mode else 'Rules'}")
            else:
                print("⚠️  No trained model loaded.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Virtual Mouse closed.")


if __name__ == "__main__":
    main()