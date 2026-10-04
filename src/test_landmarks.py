import cv2
import mediapipe as mp
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

VIDEO_PATH = Path(
    "data/raw/ASL_Citizen/train/014770041748456197-MOVIE.mp4"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video:", VIDEO_PATH.name)
print("FPS:", fps)
print("Total frames:", frame_count)

# --------------------------------------------------
# Detect landmarks
# --------------------------------------------------

frames_with_hands = 0
total_hands = 0
processed_frames = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frames += 1

        # OpenCV BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must increase for VIDEO mode
        timestamp_ms = int(
            (processed_frames - 1) * 1000 / fps
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Check detected hands
        if result.hand_landmarks:

            frames_with_hands += 1
            total_hands += len(result.hand_landmarks)

cap.release()

# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 50)
print("LANDMARK TEST COMPLETE")
print("=" * 50)

print("Frames processed:", processed_frames)
print("Frames with hands:", frames_with_hands)
print("Total hands detected:", total_hands)

if processed_frames > 0:
    percentage = (
        frames_with_hands / processed_frames
    ) * 100

    print(
        f"Hand detection rate: {percentage:.2f}%"
    )

if frames_with_hands > 0:
    print()
    print("SUCCESS: MediaPipe detected hands.")
else:
    print()
    print("WARNING: No hands were detected.")