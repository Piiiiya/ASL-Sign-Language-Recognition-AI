from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences_active_45"
)

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
EPOCHS = 50

NUM_CLASSES = 20
SEQUENCE_LENGTH = 45
NUM_FEATURES = 126

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING 45-FRAME DATASET")
print("=" * 80)

X_train = tf.keras.utils.numpy_array_to_dataset if False else None

import numpy as np

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")

assert X_train.shape[1:] == (
    SEQUENCE_LENGTH,
    NUM_FEATURES
)

assert X_val.shape[1:] == (
    SEQUENCE_LENGTH,
    NUM_FEATURES
)

assert X_test.shape[1:] == (
    SEQUENCE_LENGTH,
    NUM_FEATURES
)

# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 80)
print("BUILDING 45-FRAME BiLSTM MODEL")
print("=" * 80)

model = models.Sequential(
    [
        layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                NUM_FEATURES
            )
        ),

        layers.Bidirectional(
            layers.LSTM(
                128,
                return_sequences=True
            )
        ),

        layers.Dropout(0.30),

        layers.LSTM(64),

        layers.Dropout(0.30),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.20),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR
    / "asl_lstm_active_45_best.keras"
)

final_model_path = (
    MODEL_DIR
    / "asl_lstm_active_45_final.keras"
)

history_path = (
    OUTPUT_DIR
    / "training_history_active_45.json"
)

checkpoint = callbacks.ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("STARTING 45-FRAME TRAINING")
print("=" * 80)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history.history,
        f,
        indent=2
    )

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nValidation Loss: "
    f"{val_loss:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(final_model_path)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"\nBest model:\n{best_model_path}"
)

print(
    f"\nFinal model:\n{final_model_path}"
)

print(
    f"\nTraining history:\n{history_path}"
)