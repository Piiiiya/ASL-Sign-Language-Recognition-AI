import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path(
    "data/raw/ASL_Citizen/train/014770041748456197-MOVIE.mp4"
)

MODEL_PATH = Path(
    "models/hand_landmarker.task"
)

OUTPUT_PATH = Path(
    "data/processed/landmarks/train/"
    "014770041748456197-MOVIE_126.npy"
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


# ============================================================
# STORAGE
# ============================================================

all_frames = []

processed_frames = 0
frames_with_hands = 0
frames_with_two_hands = 0


# ============================================================
# CREATE LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        processed_frames += 1


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = processed_frames


        # ----------------------------------------------------
        # Detect hands
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # 126 FEATURES
        #
        # LEFT HAND  = 63
        # RIGHT HAND = 63
        #
        # TOTAL = 126
        # ====================================================

        left_hand = np.zeros(
            63,
            dtype=np.float32
        )

        right_hand = np.zeros(
            63,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Process detected hands
        # ----------------------------------------------------

        if result.hand_landmarks:

            frames_with_hands += 1


            if len(result.hand_landmarks) >= 2:

                frames_with_two_hands += 1


            for hand_index, hand in enumerate(
                result.hand_landmarks
            ):

                values = []

                for landmark in hand:

                    values.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])


                hand_array = np.array(
                    values,
                    dtype=np.float32
                )


                # ------------------------------------------------
                # Determine LEFT / RIGHT
                # ------------------------------------------------

                handedness = (
                    result.handedness[hand_index]
                )


                if handedness:

                    hand_label = (
                        handedness[0].category_name
                    )

                else:

                    hand_label = ""


                # ------------------------------------------------
                # Store in consistent position
                # ------------------------------------------------

                if hand_label == "Left":

                    left_hand = hand_array

                elif hand_label == "Right":

                    right_hand = hand_array


                else:

                    # Fallback if handedness is unavailable
                    if not np.any(left_hand):

                        left_hand = hand_array

                    elif not np.any(right_hand):

                        right_hand = hand_array


        # ----------------------------------------------------
        # Combine
        #
        # [Left 63 | Right 63]
        # ----------------------------------------------------

        frame_features = np.concatenate([
            left_hand,
            right_hand
        ])


        all_frames.append(
            frame_features
        )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# CREATE ARRAY
# ============================================================

landmarks_array = np.array(
    all_frames,
    dtype=np.float32
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT_PATH,
    landmarks_array
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("TWO-HAND LANDMARK TEST COMPLETE")
print("=" * 60)

print(
    "Video:",
    VIDEO_PATH.name
)

print(
    "Frames processed:",
    processed_frames
)

print(
    "Frames with hands:",
    frames_with_hands
)

print(
    "Frames with two hands:",
    frames_with_two_hands
)

if processed_frames > 0:

    detection_rate = (
        frames_with_hands /
        processed_frames *
        100
    )

    print(
        "Hand detection rate:",
        f"{detection_rate:.2f}%"
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


# ============================================================
# VERIFY 126 FEATURES
# ============================================================

print()

if (
    landmarks_array.ndim == 2
    and landmarks_array.shape[1] == 126
):

    print(
        "SUCCESS: 126 features confirmed."
    )

else:

    print(
        "ERROR: Expected 126 features."
    )