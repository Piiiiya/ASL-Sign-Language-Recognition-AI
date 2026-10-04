from pathlib import Path
import cv2

DATASET_DIR = Path("data/raw/ASL_Citizen")

splits = ["train", "val", "test"]

total = 0
valid = 0
invalid = 0

for split in splits:

    split_dir = DATASET_DIR / split
    videos = list(split_dir.glob("*.mp4"))

    print()
    print(f"{split.upper()}: {len(videos)} videos")

    split_valid = 0
    split_invalid = 0

    for video_path in videos:

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("INVALID:", video_path.name)
            split_invalid += 1
            invalid += 1
            cap.release()
            continue

        ret, frame = cap.read()

        if ret and frame is not None:
            split_valid += 1
            valid += 1
        else:
            print("INVALID:", video_path.name)
            split_invalid += 1
            invalid += 1

        cap.release()

    total += len(videos)

    print(f"Valid:   {split_valid}")
    print(f"Invalid: {split_invalid}")

print()
print("=" * 50)
print("VIDEO VERIFICATION COMPLETE")
print("=" * 50)
print(f"Total videos:   {total}")
print(f"Valid videos:   {valid}")
print(f"Invalid videos: {invalid}")