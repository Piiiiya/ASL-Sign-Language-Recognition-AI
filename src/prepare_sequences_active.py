from pathlib import Path
import zipfile
import csv
import json

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ZIP_PATH = Path(r"C:\Users\hasibul shaikh\Downloads\ASL_Citizen.zip")

LANDMARK_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "landmarks"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences_active"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SELECTED_SIGNS = [
    "APPLE",
    "DOG1",
    "BREAKFAST1",
    "DARK1",
    "DEAF1",
    "PARTY1",
    "DEVELOP1",
    "BEE1",
    "FINE1",
    "CHRISTMAS1",
    "HALLOWEEN1",
    "RIVER1",
    "LUNCH1",
    "HOSPITAL1",
    "MEAT1",
    "DINNER1",
    "EAT1",
    "MOVIE1",
    "SINK",
    "BASKETBALL1",
]

SEQUENCE_LENGTH = 30

# Videos with fewer than this many useful hand frames
# are considered too poor for sequence generation.
MIN_ACTIVE_FRAMES = 10


LABEL_TO_ID = {
    sign: i
    for i, sign in enumerate(SELECTED_SIGNS)
}

ID_TO_LABEL = {
    str(i): sign
    for sign, i in LABEL_TO_ID.items()
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_split_rows(zip_file, split):
    """
    Read the original ASL Citizen split CSV from the ZIP.
    """

    csv_path = f"ASL_Citizen/splits/{split}.csv"

    rows = []

    with zip_file.open(csv_path) as f:

        text_file = (
            line.decode("utf-8-sig")
            for line in f
        )

        reader = csv.DictReader(text_file)

        for row in reader:

            gloss = row["Gloss"].strip()

            if gloss in SELECTED_SIGNS:
                rows.append(row)

    return rows


def normalize_frame(frame):
    """
    Normalize one 126-feature frame.

    First 63 features:
        Left hand = 21 landmarks x XYZ

    Next 63 features:
        Right hand = 21 landmarks x XYZ

    Each detected hand is:
        1. translated relative to wrist
        2. scaled by maximum wrist-to-landmark distance
    """

    output = np.zeros(126, dtype=np.float32)

    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    left = frame[:63].reshape(21, 3)

    if not np.all(left == 0):

        wrist = left[0].copy()

        left = left - wrist

        distances = np.linalg.norm(left, axis=1)

        scale = np.max(distances)

        if scale > 1e-8:
            left = left / scale

        output[:63] = left.reshape(-1)

    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    right = frame[63:].reshape(21, 3)

    if not np.all(right == 0):

        wrist = right[0].copy()

        right = right - wrist

        distances = np.linalg.norm(right, axis=1)

        scale = np.max(distances)

        if scale > 1e-8:
            right = right / scale

        output[63:] = right.reshape(-1)

    return output


def remove_inactive_frames(data):
    """
    Remove frames where BOTH hands are missing.

    A frame is active when at least one hand is detected.
    """

    left = data[:, :63]
    right = data[:, 63:]

    left_missing = np.all(left == 0, axis=1)
    right_missing = np.all(right == 0, axis=1)

    active_mask = ~(left_missing & right_missing)

    return data[active_mask]


def resample_sequence(sequence, target_length=30):
    """
    Resample a sequence to exactly target_length frames
    using linear interpolation.
    """

    num_frames = len(sequence)

    if num_frames == target_length:
        return sequence.astype(np.float32)

    if num_frames == 1:
        return np.repeat(
            sequence,
            target_length,
            axis=0
        ).astype(np.float32)

    old_positions = np.linspace(
        0,
        num_frames - 1,
        num_frames
    )

    new_positions = np.linspace(
        0,
        num_frames - 1,
        target_length
    )

    result = np.empty(
        (target_length, sequence.shape[1]),
        dtype=np.float32
    )

    for feature_index in range(sequence.shape[1]):

        result[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature_index]
        )

    return result.astype(np.float32)


def process_split(zip_file, split):

    print("\n" + "=" * 80)
    print(f"PROCESSING {split.upper()} SPLIT")
    print("=" * 80)

    rows = load_split_rows(zip_file, split)

    print("Selected CSV rows:", len(rows))

    X = []
    y = []

    missing_landmarks = 0
    invalid_landmarks = 0
    skipped_low_quality = 0

    total_active_frames = 0
    total_original_frames = 0

    class_counts = {
        sign: 0
        for sign in SELECTED_SIGNS
    }

    for index, row in enumerate(rows, start=1):

        video_file = row["Video file"]
        gloss = row["Gloss"].strip()

        filename = Path(video_file).name

        landmark_path = (
            LANDMARK_DIR
            / split
            / (Path(filename).stem + ".npy")
        )

        if not landmark_path.exists():

            missing_landmarks += 1

            print(
                f"[MISSING] {filename}"
            )

            continue

        try:

            data = np.load(landmark_path)

        except Exception as e:

            invalid_landmarks += 1

            print(
                f"[INVALID] {filename}: {e}"
            )

            continue

        if (
            data.ndim != 2
            or data.shape[1] != 126
            or data.shape[0] == 0
        ):

            invalid_landmarks += 1

            print(
                f"[INVALID SHAPE] "
                f"{filename}: {data.shape}"
            )

            continue

        original_frames = data.shape[0]

        # ----------------------------------------------------
        # REMOVE FRAMES WHERE BOTH HANDS ARE MISSING
        # ----------------------------------------------------

        active_data = remove_inactive_frames(data)

        active_frames = len(active_data)

        total_original_frames += original_frames
        total_active_frames += active_frames

        # ----------------------------------------------------
        # SKIP EXTREMELY LOW-QUALITY VIDEOS
        # ----------------------------------------------------

        if active_frames < MIN_ACTIVE_FRAMES:

            skipped_low_quality += 1

            print(
                f"[SKIP LOW QUALITY] "
                f"{filename} | "
                f"active={active_frames}/"
                f"{original_frames}"
            )

            continue

        # ----------------------------------------------------
        # NORMALIZE EVERY ACTIVE FRAME
        # ----------------------------------------------------

        normalized = np.array(
            [
                normalize_frame(frame)
                for frame in active_data
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # RESAMPLE TO 30 FRAMES
        # ----------------------------------------------------

        sequence = resample_sequence(
            normalized,
            SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label_id = LABEL_TO_ID[gloss]

        X.append(sequence)
        y.append(label_id)

        class_counts[gloss] += 1

        if index % 50 == 0 or index == len(rows):

            print(
                f"Processed {index}/{len(rows)}"
            )

    # ========================================================
    # SAVE DATA
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    np.save(
        OUTPUT_DIR / f"X_{split}.npy",
        X
    )

    np.save(
        OUTPUT_DIR / f"y_{split}.npy",
        y
    )

    with open(
        OUTPUT_DIR / f"{split}_class_counts.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_counts,
            f,
            indent=4
        )

    print("\n" + "-" * 80)
    print(f"{split.upper()} RESULT")
    print("-" * 80)

    print(
        "Final X shape       :",
        X.shape
    )

    print(
        "Final y shape       :",
        y.shape
    )

    print(
        "Missing landmarks   :",
        missing_landmarks
    )

    print(
        "Invalid landmarks   :",
        invalid_landmarks
    )

    print(
        "Low-quality skipped :",
        skipped_low_quality
    )

    print(
        "Original frames     :",
        total_original_frames
    )

    print(
        "Active frames       :",
        total_active_frames
    )

    if total_original_frames > 0:

        active_percentage = (
            total_active_frames
            / total_original_frames
            * 100
        )

        print(
            f"Active frame ratio  : "
            f"{active_percentage:.2f}%"
        )

    print("\nClass counts:")

    for sign in SELECTED_SIGNS:

        print(
            f"  {sign:15} : "
            f"{class_counts[sign]}"
        )

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ASL ACTIVE-FRAME SEQUENCE PREPARATION")
    print("=" * 80)

    print("\nZIP:")
    print(ZIP_PATH)

    print("\nOutput:")
    print(OUTPUT_DIR)

    print("\nSelected signs:", len(SELECTED_SIGNS))

    print(
        "Sequence length:",
        SEQUENCE_LENGTH
    )

    print(
        "Minimum active frames:",
        MIN_ACTIVE_FRAMES
    )

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"ASL Citizen ZIP not found:\n{ZIP_PATH}"
        )

    # --------------------------------------------------------
    # SAVE LABEL MAPPING
    # --------------------------------------------------------

    label_mapping = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
    }

    with open(
        OUTPUT_DIR / "label_mapping.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            label_mapping,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # OPEN ZIP
    # --------------------------------------------------------

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zip_file:

        train_X, train_y = process_split(
            zip_file,
            "train"
        )

        val_X, val_y = process_split(
            zip_file,
            "val"
        )

        test_X, test_y = process_split(
            zip_file,
            "test"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Train X : {train_X.shape}"
    )

    print(
        f"Train y : {train_y.shape}"
    )

    print(
        f"Val X   : {val_X.shape}"
    )

    print(
        f"Val y   : {val_y.shape}"
    )

    print(
        f"Test X  : {test_X.shape}"
    )

    print(
        f"Test y  : {test_y.shape}"
    )

    print(
        "\nSaved to:"
    )

    print(OUTPUT_DIR)

    print("\nSUCCESS: ACTIVE-FRAME DATASET CREATED.")


if __name__ == "__main__":
    main()