from pathlib import Path
import numpy as np
import pandas as pd
import zipfile
import json


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

LANDMARK_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "landmarks"
)

ZIP_PATH = (
    Path.home()
    / "Downloads"
    / "ASL_Citizen.zip"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

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

SPLITS = ["train", "val", "test"]


# ============================================================
# CREATE LABEL MAPPING
# ============================================================

label_to_id = {
    sign: index
    for index, sign in enumerate(SELECTED_SIGNS)
}

id_to_label = {
    index: sign
    for sign, index in label_to_id.items()
}

print("=" * 70)
print("ASL CITIZEN - SEQUENCE PREPARATION")
print("=" * 70)

print("\nSelected signs:")
for index, sign in enumerate(SELECTED_SIGNS):
    print(f"{index:2d} -> {sign}")


# ============================================================
# SAVE LABEL MAPPING
# ============================================================

with open(
    OUTPUT_DIR / "label_mapping.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "label_to_id": label_to_id,
            "id_to_label": {
                str(k): v
                for k, v in id_to_label.items()
            }
        },
        f,
        indent=4
    )


# ============================================================
# LOAD SPLIT CSV FROM ZIP
# ============================================================

def load_split_csv(split):
    """
    Read train/val/test CSV directly from ASL_Citizen.zip.
    """

    csv_name = f"ASL_Citizen/splits/{split}.csv"

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


# ============================================================
# CREATE VIDEO -> LABEL MAPPING
# ============================================================

def create_video_label_mapping(split):
    """
    Creates:

        video_filename -> sign label
    """

    df = load_split_csv(split)

    # Keep only our selected signs
    df = df[df["Gloss"].isin(SELECTED_SIGNS)].copy()

    mapping = {}

    for _, row in df.iterrows():

        video_file = Path(str(row["Video file"])).name
        gloss = str(row["Gloss"])

        mapping[video_file] = gloss

    return mapping


# ============================================================
# LANDMARK NORMALIZATION
# ============================================================

def normalize_hand(hand):
    """
    Normalize one hand.

    Input:
        (63,)

    Reshape:
        (21, 3)

    Steps:
        1. Move wrist to origin
        2. Scale using maximum distance from wrist

    Returns:
        (63,)
    """

    hand = hand.reshape(21, 3).astype(np.float32)

    # If the hand is missing, keep zeros
    if np.allclose(hand, 0):
        return np.zeros(63, dtype=np.float32)

    # Wrist = landmark 0
    wrist = hand[0].copy()

    # Move wrist to origin
    hand = hand - wrist

    # Calculate distance of every landmark from wrist
    distances = np.linalg.norm(hand, axis=1)

    scale = np.max(distances)

    # Avoid division by zero
    if scale < 1e-6:
        return np.zeros(63, dtype=np.float32)

    hand = hand / scale

    return hand.reshape(63).astype(np.float32)


# ============================================================
# NORMALIZE FULL FRAME
# ============================================================

def normalize_sequence(sequence):
    """
    Input:
        (frames, 126)

    Output:
        (frames, 126)
    """

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for frame_index in range(sequence.shape[0]):

        # Left hand = first 63
        left_hand = sequence[
            frame_index, :63
        ]

        # Right hand = second 63
        right_hand = sequence[
            frame_index, 63:
        ]

        normalized[
            frame_index, :63
        ] = normalize_hand(left_hand)

        normalized[
            frame_index, 63:
        ] = normalize_hand(right_hand)

    return normalized


# ============================================================
# TEMPORAL RESAMPLING
# ============================================================

def resample_sequence(sequence, target_length=30):
    """
    Convert any number of frames to exactly target_length.

    Example:

        (51, 126)  -> (30, 126)
        (84, 126)  -> (30, 126)
        (206,126)  -> (30, 126)
        (540,126)  -> (30, 126)

    Uses linear interpolation across the entire sequence.
    """

    original_length = sequence.shape[0]

    if original_length == target_length:
        return sequence.astype(np.float32)

    if original_length == 1:
        return np.repeat(
            sequence,
            target_length,
            axis=0
        ).astype(np.float32)

    old_indices = np.linspace(
        0,
        original_length - 1,
        original_length
    )

    new_indices = np.linspace(
        0,
        original_length - 1,
        target_length
    )

    output = np.zeros(
        (target_length, sequence.shape[1]),
        dtype=np.float32
    )

    for feature_index in range(sequence.shape[1]):

        output[:, feature_index] = np.interp(
            new_indices,
            old_indices,
            sequence[:, feature_index]
        )

    return output.astype(np.float32)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    print("\n" + "-" * 70)
    print(f"PROCESSING {split.upper()}")
    print("-" * 70)

    split_landmark_dir = LANDMARK_DIR / split

    if not split_landmark_dir.exists():
        raise FileNotFoundError(
            f"Landmark directory not found:\n"
            f"{split_landmark_dir}"
        )

    video_label_mapping = create_video_label_mapping(split)

    print(
        f"Selected videos in CSV : "
        f"{len(video_label_mapping)}"
    )

    landmark_files = sorted(
        split_landmark_dir.glob("*.npy")
    )

    print(
        f"Landmark files found   : "
        f"{len(landmark_files)}"
    )

    X = []
    y = []

    missing_labels = []
    invalid_files = []

    for index, landmark_path in enumerate(
        landmark_files,
        start=1
    ):

        video_name = landmark_path.stem + ".mp4"

        # ----------------------------------------------------
        # Get label
        # ----------------------------------------------------

        if video_name not in video_label_mapping:

            missing_labels.append(video_name)
            continue

        gloss = video_label_mapping[video_name]

        # ----------------------------------------------------
        # Load landmark sequence
        # ----------------------------------------------------

        try:

            sequence = np.load(
                landmark_path,
                allow_pickle=False
            )

        except Exception as e:

            invalid_files.append(
                (video_name, str(e))
            )
            continue

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if (
            sequence.ndim != 2
            or sequence.shape[1] != 126
            or sequence.shape[0] == 0
        ):

            invalid_files.append(
                (
                    video_name,
                    f"Invalid shape: {sequence.shape}"
                )
            )
            continue

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        sequence = normalize_sequence(sequence)

        # ----------------------------------------------------
        # Resample to 30 frames
        # ----------------------------------------------------

        sequence = resample_sequence(
            sequence,
            SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # Add to dataset
        # ----------------------------------------------------

        X.append(sequence)
        y.append(label_to_id[gloss])

        # Progress
        if index % 25 == 0 or index == len(landmark_files):

            print(
                f"Processed {index:3d}/"
                f"{len(landmark_files)}"
            )

    # ========================================================
    # Convert to NumPy arrays
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print(f"{split.upper()} SUMMARY")
    print("-" * 70)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) > 0:

        print(
            "Expected sequence shape:",
            (len(X), SEQUENCE_LENGTH, 126)
        )

        print(
            "X dtype:",
            X.dtype
        )

        print(
            "X min:",
            np.min(X)
        )

        print(
            "X max:",
            np.max(X)
        )

        print(
            "X mean:",
            np.mean(X)
        )

    print(
        "Missing labels:",
        len(missing_labels)
    )

    print(
        "Invalid files:",
        len(invalid_files)
    )

    # ========================================================
    # SAVE
    # ========================================================

    np.save(
        OUTPUT_DIR / f"X_{split}.npy",
        X
    )

    np.save(
        OUTPUT_DIR / f"y_{split}.npy",
        y
    )

    # ========================================================
    # SAVE CLASS COUNTS
    # ========================================================

    class_counts = {}

    for class_id in range(len(SELECTED_SIGNS)):

        count = int(
            np.sum(y == class_id)
        )

        class_counts[
            SELECTED_SIGNS[class_id]
        ] = count

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

    # ========================================================
    # RETURN
    # ========================================================

    return X, y, missing_labels, invalid_files


# ============================================================
# PROCESS ALL SPLITS
# ============================================================

all_results = {}

for split in SPLITS:

    result = process_split(split)

    all_results[split] = result


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SEQUENCE VERIFICATION")
print("=" * 70)

total_samples = 0

for split in SPLITS:

    X, y, missing_labels, invalid_files = (
        all_results[split]
    )

    print("\n" + split.upper())

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim == 3:

        print(
            "Sequence length:",
            X.shape[1]
        )

        print(
            "Features:",
            X.shape[2]
        )

    total_samples += len(X)


print("\n" + "-" * 70)

print(
    "Total samples:",
    total_samples
)

print(
    "Expected samples:",
    760
)

print(
    "Expected sequence:",
    "(30, 126)"
)


# ============================================================
# FINAL SUCCESS CHECK
# ============================================================

success = True

for split in SPLITS:

    X, y, missing_labels, invalid_files = (
        all_results[split]
    )

    if X.ndim != 3:
        success = False

    elif X.shape[1:] != (30, 126):
        success = False

    if len(missing_labels) > 0:
        success = False

    if len(invalid_files) > 0:
        success = False


if total_samples != 760:
    success = False


print("\n" + "=" * 70)

if success:

    print(
        "SUCCESS: ALL 760 SEQUENCES PREPARED"
    )

    print(
        "SUCCESS: EVERY SEQUENCE HAS 30 FRAMES"
    )

    print(
        "SUCCESS: EVERY FRAME HAS 126 FEATURES"
    )

else:

    print(
        "WARNING: SEQUENCE PREPARATION NEEDS REVIEW"
    )

print("=" * 70)