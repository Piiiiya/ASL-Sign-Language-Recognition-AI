from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras


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
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
EPOCHS = 50

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126
NUM_CLASSES = 20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ASL CITIZEN - LSTM TRAINING")
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
# VERIFY DATA
# ============================================================

print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


assert X_train.shape == (383, 30, 126)
assert y_train.shape == (383,)

assert X_val.shape == (79, 30, 126)
assert y_val.shape == (79,)

assert X_test.shape == (298, 30, 126)
assert y_test.shape == (298,)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(
    DATA_DIR / "label_mapping.json",
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)

id_to_label = {
    int(k): v
    for k, v in label_mapping["id_to_label"].items()
}


print("\nClasses:")

for class_id in range(NUM_CLASSES):
    print(
        f"{class_id:2d} -> "
        f"{id_to_label[class_id]}"
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding model...")

model = keras.Sequential(
    [

        keras.layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                NUM_FEATURES
            )
        ),

        keras.layers.Bidirectional(
            keras.layers.LSTM(
                128,
                return_sequences=True
            )
        ),

        keras.layers.Dropout(0.30),

        keras.layers.LSTM(
            64,
            return_sequences=False
        ),

        keras.layers.Dropout(0.30),

        keras.layers.Dense(
            64,
            activation="relu"
        ),

        keras.layers.Dropout(0.20),

        keras.layers.Dense(
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
# CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR
    / "asl_lstm_best.keras"
)


callbacks = [

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=1e-6,

        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(

        filepath=best_model_path,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    f"Epochs    : {EPOCHS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Train     : {len(X_train)} samples"
)

print(
    f"Validation: {len(X_val)} samples"
)

print("=" * 70)


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = (
    OUTPUT_DIR
    / "training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            key: [
                float(value)
                for value in values
            ]
            for key, values
            in history.history.items()
        },
        f,
        indent=4
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(
    f"Validation Loss     : {val_loss:.4f}"
)

print(
    f"Validation Accuracy : {val_accuracy:.4f}"
)

print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / "asl_lstm_final.keras"
)

model.save(final_model_path)


# ============================================================
# TRAINING SUMMARY
# ============================================================

best_epoch = int(
    np.argmax(
        history.history["val_accuracy"]
    )
) + 1

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Epochs completed    : "
    f"{len(history.history['loss'])}"
)

print(
    f"Best epoch          : "
    f"{best_epoch}"
)

print(
    f"Best validation acc : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nBest model:"
)

print(best_model_path)

print(
    "\nFinal model:"
)

print(final_model_path)

print(
    "\nTraining history:"
)

print(history_path)

print("=" * 70)