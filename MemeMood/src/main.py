"""
main.py
--------
Entry point for MemeMood. Opens the webcam, runs the detection ->
classification -> meme-mapping pipeline on each frame, and displays the
result in a live window.

Controls:
    q  -  quit
"""

import cv2
import time

from face_detection import FaceDetector
from emotion_detection import EmotionClassifier
from meme_identifier import MemeIdentifier

WINDOW_NAME = "MemeMood"
BOX_COLOR = (172, 59, 108)      # BGR
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, box, label, confidence):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

    caption = f"{label}  ({confidence * 100:.0f}%)"
    (tw, th), _ = cv2.getTextSize(caption, FONT, 0.7, 2)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 10, y), BOX_COLOR, -1)
    cv2.putText(frame, caption, (x + 5, y - 8), FONT, 0.7, TEXT_COLOR, 2)


def draw_meme_thumbnail(frame, meme_path, size=160):
    meme_img = cv2.imread(meme_path)
    if meme_img is None:
        return frame

    meme_img = cv2.resize(meme_img, (size, size))
    h, w = frame.shape[:2]
    frame[10:10 + size, w - size - 10: w - 10] = meme_img
    return frame


def main():
    detector = FaceDetector()
    classifier = EmotionClassifier()
    meme_picker = MemeIdentifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access the webcam. Check camera permissions/index.")

    print("MemeMood is running. Press 'q' in the video window to quit.")

    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural, selfie-style view
            box = detector.primary_face(frame)

            if box is not None:
                face_crop = detector.crop(frame, box, margin=0.15)
                label, confidence, _ = classifier.predict(face_crop)
                draw_overlay(frame, box, label, confidence)

                meme_path = meme_picker.get_meme(label)
                if meme_path:
                    frame = draw_meme_thumbnail(frame, meme_path)

            # FPS counter (useful during evaluation/testing)
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"{fps:4.1f} FPS", (10, 25), FONT, 0.6, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
