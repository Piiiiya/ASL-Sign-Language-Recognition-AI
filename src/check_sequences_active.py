from pathlib import Path
import numpy as np
import json

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences_active"
)

EXPECTED_FEATURES = 126
EXPECTED_FRAMES = 30
EXPECTED_CLASSES = 20

SPLITS = ["train", "val", "test"]


def check_split(split):

    print("\n" + "=" * 75)
    print(f"{split.upper()} CHECK")
    print("=" * 75)

    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Shape check
    shape_ok = (
        X.ndim == 3
        and X.shape[1] == EXPECTED_FRAMES
        and X.shape[2] == EXPECTED_FEATURES
    )

    print("Shape valid:", shape_ok)

    # Length check
    length_ok = len(X) == len(y)

    print("X/y length match:", length_ok)

    # Data type
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)

    # NaN
    nan_found = np.isnan(X).any()

    print("NaN found:", nan_found)

    # Inf
    inf_found = np.isinf(X).any()

    print("Inf found:", inf_found)

    # Range
    print("X minimum:", float(X.min()))
    print("X maximum:", float(X.max()))

    # Labels
    unique_labels = np.unique(y)

    print("Unique labels:", unique_labels.tolist())
    print("Number of classes:", len(unique_labels))

    labels_ok = (
        unique_labels.min() >= 0
        and unique_labels.max() < EXPECTED_CLASSES
        and len(unique_labels) == EXPECTED_CLASSES
    )

    print("Labels valid:", labels_ok)

    # Class counts
    print("\nClass distribution:")

    counts = np.bincount(
        y,
        minlength=EXPECTED_CLASSES
    )

    with open(
        DATA_DIR / "label_mapping.json",
        "r",
        encoding="utf-8"
    ) as f:
        mapping = json.load(f)

    id_to_label = mapping["id_to_label"]

    for class_id in range(EXPECTED_CLASSES):

        label = id_to_label[str(class_id)]

        print(
            f"  {class_id:2d} "
            f"{label:15} : "
            f"{counts[class_id]}"
        )

    # Final result
    success = (
        shape_ok
        and length_ok
        and not nan_found
        and not inf_found
        and labels_ok
    )

    print("\nResult:", "PASS" if success else "FAIL")

    return success


def main():

    print("=" * 75)
    print("ASL ACTIVE SEQUENCE DATASET VERIFICATION")
    print("=" * 75)

    all_passed = True

    for split in SPLITS:

        passed = check_split(split)

        if not passed:
            all_passed = False

    print("\n" + "=" * 75)
    print("FINAL RESULT")
    print("=" * 75)

    if all_passed:
        print("SUCCESS: ACTIVE SEQUENCE DATASET IS READY FOR TRAINING.")
    else:
        print("ERROR: DATASET VERIFICATION FAILED.")


if __name__ == "__main__":
    main()