"""
train_model.py
----------------
Minimal training script for the emotion-recognition CNN described in the
project report. Expects data organised as:

    dataset/
        train/
            angry/ *.jpg
            disgust/ *.jpg
            fear/ *.jpg
            happy/ *.jpg
            neutral/ *.jpg
            sad/ *.jpg
            surprise/ *.jpg
        test/
            (same structure)

This layout matches how FER-2013 is commonly distributed as image folders.
Adjust DATASET_DIR if your copy is organised differently.

Usage:
    python src/train_model.py
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "emotion_model.h5")

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(48, 48, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_generators():
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    test_flow = test_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "test"),
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    return train_flow, test_flow


def main():
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)

    train_flow, test_flow = build_generators()
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        train_flow,
        validation_data=test_flow,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"Best model saved to: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
