from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126
NUM_CLASSES = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ASL CITIZEN - LSTM MODEL SETUP")
print("=" * 70)

print("\nLoading datasets...")

X_train = np.load(
    DATA_DIR / "X_train.npy",
    allow_pickle=False
)

y_train = np.load(
    DATA_DIR / "y_train.npy",
    allow_pickle=False
)

X_val = np.load(
    DATA_DIR / "X_val.npy",
    allow_pickle=False
)

y_val = np.load(
    DATA_DIR / "y_val.npy",
    allow_pickle=False
)

X_test = np.load(
    DATA_DIR / "X_test.npy",
    allow_pickle=False
)

y_test = np.load(
    DATA_DIR / "y_test.npy",
    allow_pickle=False
)


# ============================================================
# PRINT DATA SHAPES
# ============================================================

print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(
    DATA_DIR / "label_mapping.json",
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)

label_to_id = label_mapping["label_to_id"]


# ============================================================
# MODEL
# ============================================================

print("\nBuilding Bidirectional LSTM model...")


model = keras.Sequential(
    [

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                NUM_FEATURES
            )
        ),

        # ----------------------------------------------------
        # First Bidirectional LSTM
        # ----------------------------------------------------

        layers.Bidirectional(
            layers.LSTM(
                128,
                return_sequences=True
            )
        ),

        layers.Dropout(0.30),

        # ----------------------------------------------------
        # Second LSTM
        # ----------------------------------------------------

        layers.LSTM(
            64,
            return_sequences=False
        ),

        layers.Dropout(0.30),

        # ----------------------------------------------------
        # Dense layer
        # ----------------------------------------------------

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.20),

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ]
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# SAVE INITIAL MODEL
# ============================================================

model_path = MODEL_DIR / "asl_lstm_v1.keras"

model.save(model_path)

print("\n")
print("=" * 70)

print(
    "SUCCESS: MODEL CREATED"
)

print(
    "Model saved to:"
)

print(model_path)

print("=" * 70)