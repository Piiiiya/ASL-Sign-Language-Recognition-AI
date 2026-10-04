import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "ASL_Citizen"
    / "train"
    / "014770041748456197-MOVIE.mp4"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "movie_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_numbers = [
    0,
    total_frames // 4,
    total_frames // 2,
    (3 * total_frames) // 4,
    total_frames - 1,
]

print("Video:", VIDEO_PATH.name)
print("Total frames:", total_frames)
print("Extracting:", frame_numbers)

for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print(f"FAILED: frame {frame_number}")
        continue

    output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"
    cv2.imwrite(str(output_path), frame)

    print("Saved:", output_path)

cap.release()

print("\nDone.")