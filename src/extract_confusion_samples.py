from pathlib import Path
import cv2
import csv

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

ZIP_PATH = Path(
    r"C:\Users\hasibul shaikh\Downloads\ASL_Citizen.zip"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "confusion_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SIGNS WE WANT TO INSPECT
# ============================================================

SIGNS = [
    "BREAKFAST1",
    "BEE1",
    "CHRISTMAS1",
    "MOVIE1",
    "SINK",
    "DINNER1",
    "EAT1",
    "RIVER1",
    "LUNCH1",
]

# Number of videos to sample per sign
VIDEOS_PER_SIGN = 3

# Number of frames to extract from each video
FRAMES_PER_VIDEO = 5

# ============================================================
# READ ZIP FILE
# ============================================================

import zipfile

print("=" * 80)
print("EXTRACTING CONFUSION SAMPLES")
print("=" * 80)

print("\nZIP file:")
print(ZIP_PATH)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")

# ============================================================
# READ TRAINING CSV
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    csv_path = "ASL_Citizen/splits/train.csv"

    print("\nReading:")
    print(csv_path)

    with z.open(csv_path) as f:
        reader = csv.DictReader(
            line.decode("utf-8") for line in f
        )

        rows = list(reader)

print(f"Total training rows: {len(rows)}")

# ============================================================
# SELECT VIDEOS
# ============================================================

selected = {}

for sign in SIGNS:
    matching = [
        row
        for row in rows
        if row["Gloss"].strip() == sign
    ]

    selected[sign] = matching[:VIDEOS_PER_SIGN]

    print(
        f"{sign:15s} -> "
        f"{len(matching)} available, "
        f"{len(selected[sign])} selected"
    )

# ============================================================
# EXTRACT FRAMES
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    for sign, videos in selected.items():

        sign_dir = OUTPUT_DIR / sign
        sign_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 80)
        print(f"Processing: {sign}")
        print("-" * 80)

        for video_number, row in enumerate(videos, start=1):

            video_file = row["Video file"]

            # Video file inside ZIP
            zip_video_path = (
                "ASL_Citizen/videos/" + video_file
            )

            print(
                f"\nVideo {video_number}: "
                f"{video_file}"
            )

            # Read video from ZIP into memory
            video_bytes = z.read(zip_video_path)

            # Temporary video file
            temp_video = OUTPUT_DIR / "_temp_video.mp4"

            with open(temp_video, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(str(temp_video))

            if not cap.isOpened():
                print("  ERROR: Could not open video")
                continue

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0:
                print("  ERROR: Invalid frame count")
                cap.release()
                continue

            print(f"  Frames: {total_frames}")
            print(f"  FPS: {fps:.2f}")

            # Select evenly spaced frames
            frame_indices = [
                int(i * (total_frames - 1) / (FRAMES_PER_VIDEO - 1))
                for i in range(FRAMES_PER_VIDEO)
            ]

            video_dir = (
                sign_dir /
                f"video_{video_number:02d}"
            )

            video_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for frame_number, frame_index in enumerate(
                frame_indices
            ):

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index
                )

                success, frame = cap.read()

                if not success:
                    print(
                        f"  Could not read frame "
                        f"{frame_index}"
                    )
                    continue

                output_file = (
                    video_dir /
                    f"frame_{frame_number:02d}.jpg"
                )

                cv2.imwrite(
                    str(output_file),
                    frame
                )

            cap.release()

    # Remove temporary video
    if temp_video.exists():
        temp_video.unlink()

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print("\nSamples saved to:")

print(OUTPUT_DIR)