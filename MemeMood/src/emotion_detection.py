"""
emotion_detection.py
---------------------
Loads a trained CNN and classifies a cropped face image into one of seven
emotion categories: angry, disgust, fear, happy, sad, surprise, neutral.

Train your own model on FER-2013 (or a similar dataset) and place the
resulting file at models/emotion_model.h5 -- see README.md for a minimal
training script outline.
"""

import os
import numpy as np
import cv2

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "emotion_model.h5"
)


class EmotionClassifier:
    """Wraps a Keras/TensorFlow emotion-classification model."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH, input_size: tuple = (48, 48)):
        self.input_size = input_size
        self.model = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self._model_path):
            print(
                f"[emotion_detection] WARNING: no model found at '{self._model_path}'.\n"
                "                    Running in stub mode -- predictions will be random.\n"
                "                    Train a model and place it at models/emotion_model.h5."
            )
            return

        # Imported lazily so the rest of the pipeline can run/be linted
        # even in environments where TensorFlow is not installed yet.
        from tensorflow.keras.models import load_model
        self.model = load_model(self._model_path)

    def preprocess(self, face_bgr):
        """Convert a cropped BGR face image into the tensor shape the model expects."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        tensor = np.expand_dims(normalized, axis=(0, -1))  # (1, H, W, 1)
        return tensor

    def predict(self, face_bgr):
        """
        Classify a cropped face image.

        Returns:
            tuple[str, float, np.ndarray]: (predicted_label, confidence, full_probabilities)
        """
        tensor = self.preprocess(face_bgr)

        if self.model is None:
            # Stub mode: return a uniform-ish random distribution so the rest
            # of the pipeline can still be exercised without a trained model.
            probs = np.random.dirichlet(np.ones(len(EMOTION_LABELS)))
        else:
            probs = self.model.predict(tensor, verbose=0)[0]

        top_idx = int(np.argmax(probs))
        return EMOTION_LABELS[top_idx], float(probs[top_idx]), probs
