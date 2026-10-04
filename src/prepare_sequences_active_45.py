from pathlib import Path
import json
import numpy as np
import csv
import zipfile

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ZIP_PATH = Path(
    r"C:\Users\hasibul shaikh\Downloads\ASL_Citizen.zip"
)

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
    / "sequences_active_45"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 45
MIN_ACTIVE_FRAMES = 10

SIGNS = [
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

LABEL_TO_ID = {
    sign: i
    for i, sign in enumerate(SIGNS)
}

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_frame(frame):
    """
    Normalize each hand independently.

    Input:
        126 values = left hand 63 + right hand 63

    Output:
        normalized 126 values
    """

    frame = frame.reshape(2, 21, 3).copy()

    for hand_index in range(2):

        hand = frame[hand_index]

        # Check if this hand is missing
        if np.allclose(hand, 0):
            continue

        # Wrist is landmark 0
        wrist = hand[0].copy()

        # Move wrist to origin
        hand -= wrist

        # Scale by maximum distance from wrist
        distances = np.linalg.norm(
            hand,
            axis=1
        )

        max_distance = distances.max()

        if max_distance > 1e-8:
            hand /= max_distance

        frame[hand_index] = hand

    return frame.reshape(126)


# ============================================================
# RESAMPLING
# ============================================================

def resample_sequence(sequence, target_length):
    """
    Resample an arbitrary number of frames
    to exactly target_length frames.
    """

    old_length = len(sequence)

    if old_length == target_length:
        return sequence.astype(np.float32)

    old_positions = np.linspace(
        0.0,
        1.0,
        old_length
    )

    new_positions = np.linspace(
        0.0,
        1.0,
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

    return result


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):

    csv_path = (
        "ASL_Citizen/splits/"
        f"{split_name}.csv"
    )

    print("\n" + "=" * 80)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 80)

    rows = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        with z.open(csv_path) as f:

            reader = csv.DictReader(
                line.decode("utf-8")
                for line in f
            )

            for row in reader:

                gloss = row["Gloss"].strip()

                if gloss in LABEL_TO_ID:
                    rows.append(row)

    print(f"Selected videos: {len(rows)}")

    X = []
    y = []

    skipped = 0
    missing_landmarks = 0
    invalid_landmarks = 0

    total_original_frames = 0
    total_active_frames = 0

    for index, row in enumerate(rows, start=1):

        video_file = row["Video file"]
        gloss = row["Gloss"].strip()

        landmark_file = (
            LANDMARK_DIR
            / split_name
            / (
                Path(video_file).stem
                + ".npy"
            )
        )

        if not landmark_file.exists():

            missing_landmarks += 1

            print(
                f"[{index}/{len(rows)}] "
                f"Missing: {landmark_file.name}"
            )

            continue

        landmarks = np.load(
            landmark_file
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if (
            landmarks.ndim != 2
            or landmarks.shape[1] != 126
            or not np.isfinite(landmarks).all()
        ):

            invalid_landmarks += 1

            print(
                f"[{index}/{len(rows)}] "
                f"Invalid: {landmark_file.name}"
            )

            continue

        total_original_frames += len(landmarks)

        # ----------------------------------------------------
        # Remove frames where BOTH hands are missing
        # ----------------------------------------------------

        reshaped = landmarks.reshape(
            -1,
            2,
            63
        )

        left = reshaped[:, 0]
        right = reshaped[:, 1]

        left_present = ~np.all(
            np.isclose(left, 0),
            axis=1
        )

        right_present = ~np.all(
            np.isclose(right, 0),
            axis=1
        )

        active_mask = (
            left_present
            | right_present
        )

        active = landmarks[active_mask]

        total_active_frames += len(active)

        # ----------------------------------------------------
        # Minimum active frames
        # ----------------------------------------------------

        if len(active) < MIN_ACTIVE_FRAMES:

            skipped += 1

            continue

        # ----------------------------------------------------
        # Normalize every active frame
        # ----------------------------------------------------

        normalized = np.array(
            [
                normalize_frame(frame)
                for frame in active
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Resample to 45 frames
        # ----------------------------------------------------

        sequence = resample_sequence(
            normalized,
            SEQUENCE_LENGTH
        )

        X.append(sequence)

        y.append(
            LABEL_TO_ID[gloss]
        )

    # ========================================================
    # SAVE
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
        OUTPUT_DIR
        / f"X_{split_name}.npy",
        X
    )

    np.save(
        OUTPUT_DIR
        / f"y_{split_name}.npy",
        y
    )

    # ========================================================
    # CLASS COUNTS
    # ========================================================

    class_counts = {}

    for sign, label_id in LABEL_TO_ID.items():

        class_counts[sign] = int(
            np.sum(y == label_id)
        )

    with open(
        OUTPUT_DIR
        / f"class_counts_{split_name}.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_counts,
            f,
            indent=2
        )

    print("\nResults:")
    print(
        f"X_{split_name}: {X.shape}"
    )

    print(
        f"y_{split_name}: {y.shape}"
    )

    print(
        f"Original frames: "
        f"{total_original_frames}"
    )

    print(
        f"Active frames: "
        f"{total_active_frames}"
    )

    if total_original_frames > 0:

        ratio = (
            total_active_frames
            / total_original_frames
            * 100
        )

        print(
            f"Active ratio: "
            f"{ratio:.2f}%"
        )

    print(
        f"Skipped (<{MIN_ACTIVE_FRAMES} active): "
        f"{skipped}"
    )

    print(
        f"Missing landmarks: "
        f"{missing_landmarks}"
    )

    print(
        f"Invalid landmarks: "
        f"{invalid_landmarks}"
    )

    print("\nClass counts:")

    for sign, count in class_counts.items():

        print(
            f"  {sign:15s}: {count}"
        )

    return X, y


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("ACTIVE 45-FRAME SEQUENCE PREPARATION")
print("=" * 80)

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"ZIP not found: {ZIP_PATH}"
    )

print(
    f"\nSequence length: "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Minimum active frames: "
    f"{MIN_ACTIVE_FRAMES}"
)

for split in [
    "train",
    "val",
    "test"
]:

    process_split(split)

# ============================================================
# SAVE LABEL MAPPING
# ============================================================

label_mapping = {
    "label_to_id": LABEL_TO_ID,
    "id_to_label": {
        str(v): k
        for k, v in LABEL_TO_ID.items()
    }
}

with open(
    OUTPUT_DIR / "label_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        label_mapping,
        f,
        indent=2
    )

print("\n" + "=" * 80)
print("45-FRAME DATASET PREPARATION COMPLETE")
print("=" * 80)

print("\nOutput:")
print(OUTPUT_DIR)