"""
face_detection.py
------------------
Locates the primary face in a video frame and returns its bounding box.

Default backend: OpenCV's Haar Cascade classifier (fast, dependency-free).
This module is intentionally isolated from the rest of the pipeline so the
detector can be swapped (e.g. for OpenCV DNN or MediaPipe) without touching
any other file.
"""

import cv2
import os

# Path to the bundled Haar Cascade XML that ships with opencv-python.
_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


class FaceDetector:
    """Wraps a face-detection backend behind a single, simple interface."""

    def __init__(self, cascade_path: str = _CASCADE_PATH, scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size: tuple = (60, 60)):
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise IOError(f"Could not load Haar Cascade from: {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame):
        """
        Detect faces in a BGR frame.

        Returns:
            list[tuple[int, int, int, int]]: bounding boxes as (x, y, w, h),
            sorted by area (largest/primary face first).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves robustness in low/uneven light

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
        return faces

    def primary_face(self, frame):
        """Return only the largest detected face, or None if no face is found."""
        faces = self.detect(frame)
        return faces[0] if len(faces) > 0 else None

    @staticmethod
    def crop(frame, box, margin: float = 0.0):
        """Crop a face region from the frame, with an optional margin (fraction of box size)."""
        x, y, w, h = box
        mx, my = int(w * margin), int(h * margin)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(frame.shape[1], x + w + mx)
        y2 = min(frame.shape[0], y + h + my)

        return frame[y1:y2, x1:x2]
