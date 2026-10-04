from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences_active"
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
SEQUENCE_LENGTH = 30
NUM_FEATURES = 126

LEARNING_RATE = 0.001


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ACTIVE-FRAME ASL BiLSTM TRAINING")
print("=" * 80)

X_train = __import__("numpy").load(
    DATA_DIR / "X_train.npy"
)

y_train = __import__("numpy").load(
    DATA_DIR / "y_train.npy"
)

X_val = __import__("numpy").load(
    DATA_DIR / "X_val.npy"
)

y_val = __import__("numpy").load(
    DATA_DIR / "y_val.npy"
)

X_test = __import__("numpy").load(
    DATA_DIR / "X_test.npy"
)

y_test = __import__("numpy").load(
    DATA_DIR / "y_test.npy"
)


print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# VALIDATION
# ============================================================

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

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)

print("\nDataset verification: PASSED")


# ============================================================
# MODEL
# ============================================================

model = keras.Sequential(
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
        ),
    ]
)


# ============================================================
# COMPILE
# ============================================================

optimizer = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR
    / "asl_lstm_active_best.keras"
)

final_model_path = (
    MODEL_DIR
    / "asl_lstm_active_final.keras"
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
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
print("STARTING TRAINING")
print("=" * 80)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

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

history_path = (
    OUTPUT_DIR
    / "training_history_active.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history.history,
        f,
        indent=4
    )


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION EVALUATION")
print("=" * 80)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(
    f"Validation Loss     : {val_loss:.4f}"
)

print(
    f"Validation Accuracy : {val_accuracy * 100:.2f}%"
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("TEST EVALUATION")
print("=" * 80)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    final_model_path
)


print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)

print(
    "Best checkpoint:",
    best_model_path
)

print(
    "Final model:",
    final_model_path
)

print(
    "History:",
    history_path
)

print("\nDone.")