from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

X_TEST_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences"
    / "X_test.npy"
)

Y_TEST_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences"
    / "y_test.npy"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "asl_lstm_best.keras"
)

LABEL_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences"
    / "label_mapping.json"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ASL SIGN LANGUAGE MODEL - DETAILED EVALUATION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    X_TEST_PATH,
    Y_TEST_PATH,
    MODEL_PATH,
    LABEL_PATH,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print("FOUND:", file_path)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("\nX_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("X_test dtype:", X_test.dtype)
print("y_test dtype:", y_test.dtype)


# ============================================================
# VERIFY TEST DATA
# ============================================================

if X_test.ndim != 3:

    raise ValueError(
        f"Expected X_test to have 3 dimensions, "
        f"but got {X_test.ndim}"
    )


if X_test.shape[1:] != (30, 126):

    raise ValueError(
        f"Expected X_test shape (*, 30, 126), "
        f"but got {X_test.shape}"
    )


if len(X_test) != len(y_test):

    raise ValueError(
        "X_test and y_test contain different numbers of samples."
    )


if np.isnan(X_test).any():

    raise ValueError("X_test contains NaN values.")


if np.isinf(X_test).any():

    raise ValueError("X_test contains infinite values.")


print("\nTest data verification: PASSED")


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

print("\n" + "=" * 70)
print("LOADING LABEL MAPPING")
print("=" * 70)

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as f:

    mapping_data = json.load(f)


print("\nMapping JSON structure:")

if isinstance(mapping_data, dict):

    for key in mapping_data.keys():

        print(" -", key)


# ============================================================
# HANDLE DIFFERENT LABEL MAPPING FORMATS
# ============================================================

if "id_to_label" in mapping_data:

    label_mapping = {
        int(k): v
        for k, v in mapping_data["id_to_label"].items()
    }

elif "label_to_id" in mapping_data:

    label_mapping = {
        int(v): k
        for k, v in mapping_data["label_to_id"].items()
    }

else:

    # Fallback in case JSON is directly:
    # {"0": "APPLE", "1": "DOG1", ...}

    try:

        label_mapping = {
            int(k): v
            for k, v in mapping_data.items()
        }

    except (ValueError, TypeError):

        raise ValueError(
            "\nUnable to understand label_mapping.json.\n"
            "Expected either 'id_to_label' or 'label_to_id'."
        )


# ============================================================
# SORT CLASS IDS
# ============================================================

class_ids = sorted(label_mapping.keys())

class_names = [
    label_mapping[class_id]
    for class_id in class_ids
]


# ============================================================
# DISPLAY CLASSES
# ============================================================

print("\nNumber of classes:", len(class_names))

print("\nClass Mapping:")
print("-" * 40)

for class_id, class_name in zip(
    class_ids,
    class_names
):

    print(
        f"{class_id:2d} -> {class_name}"
    )


# ============================================================
# VERIFY LABELS
# ============================================================

unique_test_labels = sorted(
    np.unique(y_test).tolist()
)

print("\nLabels found in test set:")

print(unique_test_labels)


invalid_labels = [
    label
    for label in unique_test_labels
    if label not in class_ids
]


if invalid_labels:

    raise ValueError(
        f"Test set contains unknown labels: {invalid_labels}"
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")


# ============================================================
# DISPLAY MODEL SUMMARY
# ============================================================

print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST PREDICTIONS")
print("=" * 70)

probabilities = model.predict(
    X_test,
    batch_size=16,
    verbose=1
)


# ============================================================
# CONVERT PROBABILITIES TO CLASS IDs
# ============================================================

y_pred = np.argmax(
    probabilities,
    axis=1
)


print("\nPrediction completed.")

print("Predictions shape:", y_pred.shape)


# ============================================================
# OVERALL TEST ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("OVERALL TEST RESULT")
print("=" * 70)

print(
    f"\nTest Accuracy : {accuracy * 100:.2f}%"
)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=class_ids,
        zero_division=0
    )
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

print(
    f"{'Class':<20}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>10}"
)

print("-" * 70)


for i, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:<20}"
        f"{precision[i] * 100:>11.2f}%"
        f"{recall[i] * 100:>11.2f}%"
        f"{f1[i] * 100:>11.2f}%"
        f"{support[i]:>10}"
    )


# ============================================================
# MACRO AVERAGE
# ============================================================

macro_precision = precision.mean()

macro_recall = recall.mean()

macro_f1 = f1.mean()


# ============================================================
# WEIGHTED AVERAGE
# ============================================================

weighted_precision = np.average(
    precision,
    weights=support
)

weighted_recall = np.average(
    recall,
    weights=support
)

weighted_f1 = np.average(
    f1,
    weights=support
)


print("\n" + "=" * 70)
print("AVERAGE METRICS")
print("=" * 70)

print(
    f"\nMacro Precision : "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1-score  : "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"\nWeighted Precision : "
    f"{weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : "
    f"{weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1-score  : "
    f"{weighted_f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_ids
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows    = Actual class")
print("Columns = Predicted class\n")


# Header

print(
    " " * 18
    + " ".join(
        f"{class_id:>4}"
        for class_id in class_ids
    )
)


# Matrix rows

for i, row in enumerate(cm):

    print(
        f"{i:>3} "
        f"{class_names[i]:<13}"
        + " ".join(
            f"{value:>4}"
            for value in row
        )
    )


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)


per_class_accuracy = {}


for i, class_name in enumerate(
    class_names
):

    total = int(
        cm[i].sum()
    )

    correct = int(
        cm[i, i]
    )


    if total > 0:

        class_accuracy = (
            correct / total
        )

    else:

        class_accuracy = 0.0


    per_class_accuracy[class_name] = {

        "class_id": int(i),

        "accuracy": float(
            class_accuracy
        ),

        "accuracy_percent": float(
            class_accuracy * 100
        ),

        "correct": correct,

        "total": total
    }


    print(
        f"{class_name:<20}"
        f"{class_accuracy * 100:>8.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# MOST CONFUSED SIGN PAIRS
# ============================================================

print("\n" + "=" * 70)
print("MOST CONFUSED SIGN PAIRS")
print("=" * 70)


confusions = []


for actual_index, actual_id in enumerate(
    class_ids
):

    for predicted_index, predicted_id in enumerate(
        class_ids
    ):

        if actual_id == predicted_id:

            continue


        count = int(
            cm[
                actual_index,
                predicted_index
            ]
        )


        if count > 0:

            confusions.append(
                (
                    count,
                    class_names[actual_index],
                    class_names[predicted_index]
                )
            )


# Highest confusion first

confusions.sort(
    key=lambda x: x[0],
    reverse=True
)


if confusions:

    for (
        count,
        actual_name,
        predicted_name
    ) in confusions[:15]:

        print(
            f"{actual_name:<20}"
            f" -> "
            f"{predicted_name:<20}"
            f": {count}"
        )

else:

    print(
        "No incorrect predictions found."
    )


# ============================================================
# CORRECT / INCORRECT COUNTS
# ============================================================

correct_predictions = int(
    np.sum(
        y_test == y_pred
    )
)

incorrect_predictions = int(
    np.sum(
        y_test != y_pred
    )


)


print("\n" + "=" * 70)
print("PREDICTION COUNTS")
print("=" * 70)

print(
    f"\nCorrect predictions   : "
    f"{correct_predictions}"
)

print(
    f"Incorrect predictions : "
    f"{incorrect_predictions}"
)

print(
    f"Total test samples    : "
    f"{len(y_test)}"
)


# ============================================================
# FULL CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    zero_division=0
)


print("\n" + "=" * 70)
print("FULL CLASSIFICATION REPORT")
print("=" * 70)

print()

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ASL SIGN LANGUAGE MODEL "
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(report)


# ============================================================
# SAVE PER-CLASS ACCURACY
# ============================================================

per_class_path = (
    OUTPUT_DIR
    / "per_class_accuracy.json"
)


with open(
    per_class_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        per_class_accuracy,
        f,
        indent=4
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = (
    OUTPUT_DIR
    / "confusion_matrix.npy"
)


np.save(
    cm_path,
    cm
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_data = []


for i in range(
    len(y_test)
):

    actual_id = int(
        y_test[i]
    )

    predicted_id = int(
        y_pred[i]
    )

    confidence = float(
        probabilities[i][predicted_id]
    )


    prediction_data.append(
        {
            "sample_index": i,

            "actual_id": actual_id,

            "actual_sign":
                label_mapping[actual_id],

            "predicted_id":
                predicted_id,

            "predicted_sign":
                label_mapping[predicted_id],

            "confidence":
                confidence,

            "confidence_percent":
                confidence * 100,

            "correct":
                actual_id == predicted_id
        }
    )


predictions_path = (
    OUTPUT_DIR
    / "test_predictions.json"
)


with open(
    predictions_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        prediction_data,
        f,
        indent=4
    )


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {

    "model":
        str(MODEL_PATH),

    "test_samples":
        int(len(y_test)),

    "number_of_classes":
        int(len(class_names)),

    "test_accuracy":
        float(accuracy),

    "test_accuracy_percent":
        float(accuracy * 100),

    "macro_precision":
        float(macro_precision),

    "macro_recall":
        float(macro_recall),

    "macro_f1":
        float(macro_f1),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1),

    "correct_predictions":
        correct_predictions,

    "incorrect_predictions":
        incorrect_predictions,

    "classes":
        label_mapping
}


summary_path = (
    OUTPUT_DIR
    / "evaluation_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nTest Accuracy      : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision    : "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall       : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1-score     : "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"\nCorrect predictions   : "
    f"{correct_predictions}"
)

print(
    f"Incorrect predictions : "
    f"{incorrect_predictions}"
)


print("\nFiles saved:")

print(
    "1.",
    report_path
)

print(
    "2.",
    per_class_path
)

print(
    "3.",
    cm_path
)

print(
    "4.",
    predictions_path
)

print(
    "5.",
    summary_path
)


print("\n" + "=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "\nWe will analyze the weak signs and "
    "most-confused sign pairs before changing "
    "the model or training strategy."
)