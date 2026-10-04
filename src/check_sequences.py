from pathlib import Path
import numpy as np
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences"
)


# ============================================================
# FILES
# ============================================================

FILES = {
    "train": ("X_train.npy", "y_train.npy"),
    "val": ("X_val.npy", "y_val.npy"),
    "test": ("X_test.npy", "y_test.npy"),
}


EXPECTED = {
    "train": 383,
    "val": 79,
    "test": 298,
}

EXPECTED_CLASSES = 20
EXPECTED_FRAMES = 30
EXPECTED_FEATURES = 126


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

mapping_path = SEQUENCE_DIR / "label_mapping.json"

with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

label_to_id = mapping["label_to_id"]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ASL CITIZEN - SEQUENCE DATASET VERIFICATION")
print("=" * 70)

print("\nNumber of classes:", len(label_to_id))

print("\nClass mapping:")

for sign, class_id in label_to_id.items():
    print(f"{class_id:2} -> {sign}")


# ============================================================
# VERIFY EACH SPLIT
# ============================================================

all_success = True
total_samples = 0

for split, (x_file, y_file) in FILES.items():

    print("\n" + "-" * 70)
    print(split.upper())
    print("-" * 70)

    x_path = SEQUENCE_DIR / x_file
    y_path = SEQUENCE_DIR / y_file

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not x_path.exists():

        print("ERROR: Missing", x_file)
        all_success = False
        continue

    if not y_path.exists():

        print("ERROR: Missing", y_file)
        all_success = False
        continue

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X = np.load(
        x_path,
        allow_pickle=False
    )

    y = np.load(
        y_path,
        allow_pickle=False
    )

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    expected_x_shape = (
        EXPECTED[split],
        EXPECTED_FRAMES,
        EXPECTED_FEATURES,
    )

    expected_y_shape = (
        EXPECTED[split],
    )

    print(
        "Expected X:",
        expected_x_shape
    )

    print(
        "Expected y:",
        expected_y_shape
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if X.shape != expected_x_shape:

        print("ERROR: X shape is incorrect")
        all_success = False

    else:

        print("X shape: OK")

    if y.shape != expected_y_shape:

        print("ERROR: y shape is incorrect")
        all_success = False

    else:

        print("y shape: OK")

    # --------------------------------------------------------
    # Dtype
    # --------------------------------------------------------

    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    if X.dtype != np.float32:

        print("WARNING: X is not float32")

    # --------------------------------------------------------
    # NaN / Inf check
    # --------------------------------------------------------

    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()

    print("NaN present:", has_nan)
    print("Inf present:", has_inf)

    if has_nan or has_inf:

        print("ERROR: X contains NaN or Inf")
        all_success = False

    # --------------------------------------------------------
    # Label range
    # --------------------------------------------------------

    min_label = int(np.min(y))
    max_label = int(np.max(y))

    print(
        "Label range:",
        min_label,
        "to",
        max_label
    )

    if min_label < 0 or max_label >= EXPECTED_CLASSES:

        print("ERROR: Invalid label detected")
        all_success = False

    # --------------------------------------------------------
    # Classes present
    # --------------------------------------------------------

    unique_labels = np.unique(y)

    print(
        "Classes present:",
        len(unique_labels)
    )

    print(
        "Class IDs:",
        unique_labels.tolist()
    )

    if len(unique_labels) != EXPECTED_CLASSES:

        print(
            "WARNING: Not all 20 classes appear in this split"
        )

    # --------------------------------------------------------
    # Value range
    # --------------------------------------------------------

    print(
        "Minimum feature value:",
        float(np.min(X))
    )

    print(
        "Maximum feature value:",
        float(np.max(X))
    )

    print(
        "Mean feature value:",
        float(np.mean(X))
    )

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    total_samples += len(X)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("Total samples:", total_samples)
print("Expected samples:", 760)

print("Expected classes:", EXPECTED_CLASSES)
print("Expected frames:", EXPECTED_FRAMES)
print("Expected features:", EXPECTED_FEATURES)


# ============================================================
# FINAL RESULT
# ============================================================

if (
    all_success
    and total_samples == 760
):

    print("\nSUCCESS: SEQUENCE DATASET IS READY FOR TRAINING")

else:

    print("\nWARNING: DATASET VERIFICATION NEEDS REVIEW")

print("=" * 70)