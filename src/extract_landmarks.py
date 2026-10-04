import cv2
import mediapipe as mp
import numpy as np
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

OUTPUT_DIR = Path(
    "data/processed/landmarks/train"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = OUTPUT_DIR / "014770041748456197-MOVIE.npy"


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
# Extract landmarks
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

all_frames = []
processed_frames = 0
frames_with_hands = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frames += 1

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp
        timestamp_ms = int(
            (processed_frames - 1) * 1000 / fps
        )

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # --------------------------------------------------
        # Create 63 features
        # 21 landmarks × (x, y, z)
        # --------------------------------------------------

        frame_landmarks = np.zeros(
            63,
            dtype=np.float32
        )

        if result.hand_landmarks:

            frames_with_hands += 1

            # Use first detected hand
            hand = result.hand_landmarks[0]

            values = []

            for landmark in hand:
                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            frame_landmarks = np.array(
                values,
                dtype=np.float32
            )

        all_frames.append(frame_landmarks)


cap.release()


# --------------------------------------------------
# Convert to NumPy array
# --------------------------------------------------

landmarks_array = np.array(
    all_frames,
    dtype=np.float32
)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.save(
    OUTPUT_PATH,
    landmarks_array
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 50)
print("LANDMARK EXTRACTION COMPLETE")
print("=" * 50)

print("Video:", VIDEO_PATH.name)
print("Frames processed:", processed_frames)
print("Frames with hands:", frames_with_hands)

print(
    "Detection rate:",
    f"{frames_with_hands / processed_frames * 100:.2f}%"
)

print(
    "Output shape:",
    landmarks_array.shape
)

print(
    "Data type:",
    landmarks_array.dtype
)

print(
    "Saved to:",
    OUTPUT_PATH
)