"""
meme_identifier.py
-------------------
Maps a predicted emotion label to a meme category and returns a random
meme image from the corresponding local folder under memes/.
"""

import os
import random

_MEMES_DIR = os.path.join(os.path.dirname(__file__), "..", "memes")

# Emotion -> meme category folder name. Edit this dictionary (or load it
# from a JSON/YAML config) to change the mapping without touching any code.
EMOTION_TO_CATEGORY = {
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "angry",
    "Surprise": "surprise",
    "Fear": "fear",
    "Disgust": "disgust",
    "Neutral": "neutral",
}

_VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class MemeIdentifier:
    """Selects a meme image file that matches a predicted emotion."""

    def __init__(self, memes_dir: str = _MEMES_DIR):
        self.memes_dir = memes_dir

    def category_for(self, emotion_label: str) -> str:
        return EMOTION_TO_CATEGORY.get(emotion_label, "neutral")

    def get_meme(self, emotion_label: str):
        """
        Return a random meme file path for the given emotion, or None if the
        category folder is empty/missing (caller should show a placeholder).
        """
        category = self.category_for(emotion_label)
        folder = os.path.join(self.memes_dir, category)

        if not os.path.isdir(folder):
            return None

        candidates = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(_VALID_EXTENSIONS)
        ]

        return random.choice(candidates) if candidates else None
