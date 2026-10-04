from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "sequences_active"
MODEL_PATH = PROJECT_DIR / "models" / "asl_lstm_active_best.keras"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("ACTIVE-FRAME ASL MODEL EVALUATION")
print("=" * 80)

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

with open(DATA_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)


# Handle nested label mapping
if "id_to_label" in mapping:
    id_to_label = mapping["id_to_label"]
else:
    id_to_label = mapping


# Convert JSON keys to integers
id_to_label = {
    int(k): v
    for k, v in id_to_label.items()
}


class_names = [
    id_to_label[i]
    for i in range(len(id_to_label))
]


print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i:2d} -> {name}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING BEST MODEL")
print("=" * 80)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model:", MODEL_PATH)
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

probabilities = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

print("\nPredictions generated:", len(y_pred))


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 80)
print("OVERALL TEST RESULTS")
print("=" * 80)

print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision_macro * 100:.2f}%")
print(f"Macro Recall      : {recall_macro * 100:.2f}%")
print(f"Macro F1          : {f1_macro * 100:.2f}%")
print(f"Weighted Precision: {precision_weighted * 100:.2f}%")
print(f"Weighted Recall   : {recall_weighted * 100:.2f}%")
print(f"Weighted F1       : {f1_weighted * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(report)


# Save report
report_path = OUTPUT_DIR / "classification_report_active.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(class_names)))
)

np.save(
    OUTPUT_DIR / "confusion_matrix_active.npy",
    cm
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

per_class_accuracy = {}

for i, class_name in enumerate(class_names):

    total = cm[i].sum()

    if total == 0:
        class_accuracy = 0.0
    else:
        class_accuracy = cm[i, i] / total

    per_class_accuracy[class_name] = {
        "correct": int(cm[i, i]),
        "total": int(total),
        "accuracy": round(class_accuracy * 100, 2)
    }


print("\n" + "=" * 80)
print("PER-CLASS ACCURACY")
print("=" * 80)

for class_name, values in per_class_accuracy.items():

    print(
        f"{class_name:15s} "
        f"{values['correct']:3d}/{values['total']:3d} "
        f"= {values['accuracy']:6.2f}%"
    )


with open(
    OUTPUT_DIR / "per_class_accuracy_active.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        per_class_accuracy,
        f,
        indent=4
    )


# ============================================================
# MOST CONFUSED PAIRS
# ============================================================

confusions = []

for actual in range(len(class_names)):

    for predicted in range(len(class_names)):

        if actual == predicted:
            continue

        count = int(cm[actual, predicted])

        if count > 0:

            confusions.append(
                (
                    count,
                    class_names[actual],
                    class_names[predicted]
                )
            )


confusions.sort(reverse=True)


print("\n" + "=" * 80)
print("TOP CONFUSION PAIRS")
print("=" * 80)

for count, actual, predicted in confusions[:20]:

    print(
        f"{actual:15s} -> {predicted:15s} : {count}"
    )


# ============================================================
# CORRECT / INCORRECT
# ============================================================

correct = int(np.sum(y_test == y_pred))
incorrect = int(np.sum(y_test != y_pred))

print("\n" + "=" * 80)
print("CORRECT / INCORRECT")
print("=" * 80)

print("Correct  :", correct)
print("Incorrect:", incorrect)
print("Total    :", len(y_test))


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_data = []

for i in range(len(y_test)):

    true_id = int(y_test[i])
    predicted_id = int(y_pred[i])

    prediction_data.append(
        {
            "index": i,
            "actual_id": true_id,
            "actual_label": class_names[true_id],
            "predicted_id": predicted_id,
            "predicted_label": class_names[predicted_id],
            "confidence": round(
                float(probabilities[i][predicted_id]) * 100,
                2
            ),
            "correct": bool(true_id == predicted_id)
        }
    )


with open(
    OUTPUT_DIR / "test_predictions_active.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        prediction_data,
        f,
        indent=4
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "model": str(MODEL_PATH),
    "test_samples": int(len(y_test)),
    "accuracy": round(accuracy * 100, 2),
    "macro_precision": round(precision_macro * 100, 2),
    "macro_recall": round(recall_macro * 100, 2),
    "macro_f1": round(f1_macro * 100, 2),
    "weighted_precision": round(precision_weighted * 100, 2),
    "weighted_recall": round(recall_weighted * 100, 2),
    "weighted_f1": round(f1_weighted * 100, 2),
    "correct": correct,
    "incorrect": incorrect,
    "classes": class_names
}


with open(
    OUTPUT_DIR / "evaluation_summary_active.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)

print("\nSaved files:")

print(
    OUTPUT_DIR / "classification_report_active.txt"
)

print(
    OUTPUT_DIR / "per_class_accuracy_active.json"
)

print(
    OUTPUT_DIR / "confusion_matrix_active.npy"
)

print(
    OUTPUT_DIR / "test_predictions_active.json"
)

print(
    OUTPUT_DIR / "evaluation_summary_active.json"
)

print("\nDone.")