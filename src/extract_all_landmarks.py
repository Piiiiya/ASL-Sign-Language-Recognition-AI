from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = PROJECT_DIR / "data" / "raw" / "ASL_Citizen"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "landmarks"

MODEL_PATH = PROJECT_DIR / "models" / "hand_landmarker.task"


# ============================================================
# SETTINGS
# ============================================================

# None = process ALL frames
MAX_FRAMES = None


# ============================================================
# CREATE MEDIAPIPE LANDMARKER
# ============================================================

def create_landmarker():

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )

    return vision.HandLandmarker.create_from_options(options)


# ============================================================
# EXTRACT 126 FEATURES FROM ONE VIDEO
# ============================================================

def extract_video(video_path, output_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frames = []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    processed_frames = 0
    frames_with_hands = 0
    frames_with_two_hands = 0

    # --------------------------------------------------------
    # IMAGE MODE
    # No timestamp required
    # --------------------------------------------------------

    with create_landmarker() as landmarker:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                break

            # ------------------------------------------------
            # Optional frame limit
            # ------------------------------------------------

            if (
                MAX_FRAMES is not None
                and processed_frames >= MAX_FRAMES
            ):
                break

            # ------------------------------------------------
            # Convert BGR -> RGB
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # Create MediaPipe image
            # ------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # ------------------------------------------------
            # Detect hands
            # ------------------------------------------------

            result = landmarker.detect(
                mp_image
            )

            # ------------------------------------------------
            # Empty hand vectors
            #
            # 21 landmarks × 3 coordinates
            # = 63 features per hand
            # ------------------------------------------------

            left_hand = np.zeros(
                63,
                dtype=np.float32
            )

            right_hand = np.zeros(
                63,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Process detected hands
            # ------------------------------------------------

            if result.hand_landmarks:

                detected_hands = len(
                    result.hand_landmarks
                )

                frames_with_hands += 1

                if detected_hands >= 2:
                    frames_with_two_hands += 1

                for hand_index, hand_landmarks in enumerate(
                    result.hand_landmarks
                ):

                    # ----------------------------------------
                    # Extract X, Y, Z
                    # ----------------------------------------

                    values = []

                    for landmark in hand_landmarks:

                        values.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    landmarks = np.asarray(
                        values,
                        dtype=np.float32
                    )

                    # ----------------------------------------
                    # Safety check
                    # ----------------------------------------

                    if landmarks.shape != (63,):
                        continue

                    # ----------------------------------------
                    # Determine handedness
                    # ----------------------------------------

                    handedness_label = None

                    if (
                        result.handedness
                        and
                        hand_index < len(
                            result.handedness
                        )
                        and
                        result.handedness[
                            hand_index
                        ]
                    ):

                        handedness_label = (
                            result.handedness[
                                hand_index
                            ][0].category_name
                        )

                    # ----------------------------------------
                    # Store correct hand
                    # ----------------------------------------

                    if handedness_label == "Left":

                        left_hand = landmarks

                    elif handedness_label == "Right":

                        right_hand = landmarks

                    else:

                        # Fallback
                        if not np.any(left_hand):

                            left_hand = landmarks

                        elif not np.any(right_hand):

                            right_hand = landmarks

            # ------------------------------------------------
            # Combine both hands
            #
            # Left  = 63
            # Right = 63
            # Total = 126
            # ------------------------------------------------

            combined = np.concatenate([
                left_hand,
                right_hand
            ])

            # ------------------------------------------------
            # Final feature check
            # ------------------------------------------------

            if combined.shape != (126,):

                raise RuntimeError(
                    f"Expected 126 features, "
                    f"got {combined.shape}"
                )

            frames.append(combined)

            processed_frames += 1

    cap.release()

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    if len(frames) == 0:

        raise RuntimeError(
            "No frames were extracted"
        )

    sequence = np.asarray(
        frames,
        dtype=np.float32
    )

    # Expected:
    #
    # (number_of_frames, 126)
    #

    if sequence.ndim != 2:

        raise RuntimeError(
            f"Unexpected array shape: "
            f"{sequence.shape}"
        )

    if sequence.shape[1] != 126:

        raise RuntimeError(
            f"Expected 126 features, "
            f"got {sequence.shape}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        sequence
    )

    return {
        "frames": sequence.shape[0],
        "features": sequence.shape[1],
        "frames_with_hands": frames_with_hands,
        "frames_with_two_hands": frames_with_two_hands,
        "total_frames": total_frames
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ASL CITIZEN - 126 FEATURE LANDMARK EXTRACTION")
    print("FULL DATASET MODE")
    print("=" * 70)

    print()
    print("Project:")
    print(PROJECT_DIR)

    print()
    print("Video directory:")
    print(VIDEO_DIR)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print()
    print("MediaPipe model:")
    print(MODEL_PATH)

    print()
    print("MAX_FRAMES:")
    print(MAX_FRAMES)

    print()

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"MediaPipe model not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_videos = 0
    processed_videos = 0
    skipped_videos = 0
    failed_videos = 0

    # ========================================================
    # PROCESS TRAIN / VAL / TEST
    # ========================================================

    for split in [
        "train",
        "val",
        "test"
    ]:

        input_dir = VIDEO_DIR / split
        output_dir = OUTPUT_DIR / split

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        videos = sorted(
            input_dir.glob("*.mp4")
        )

        print()
        print("-" * 70)
        print(
            f"SPLIT: {split.upper()}"
        )
        print(
            f"Videos: {len(videos)}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Process videos
        # ----------------------------------------------------

        for index, video_path in enumerate(
            videos,
            start=1
        ):

            total_videos += 1

            # Same filename as video
            output_path = (
                output_dir /
                f"{video_path.stem}.npy"
            )

            # =================================================
            # IMPORTANT
            #
            # We DO NOT skip existing files.
            #
            # This intentionally overwrites the previous
            # 30-frame files with complete full-length files.
            # =================================================

            try:

                info = extract_video(
                    video_path,
                    output_path
                )

                processed_videos += 1

                print(
                    f"[{index}/{len(videos)}] "
                    f"OK: {video_path.name} | "
                    f"frames={info['frames']} | "
                    f"hands={info['frames_with_hands']} | "
                    f"2hands={info['frames_with_two_hands']} | "
                    f"shape=({info['frames']},126)"
                )

            except Exception as e:

                failed_videos += 1

                print(
                    f"[{index}/{len(videos)}] "
                    f"FAILED: {video_path.name}"
                )

                print(
                    "   Error:",
                    repr(e)
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Total videos encountered : "
        f"{total_videos}"
    )

    print(
        f"Processed                 : "
        f"{processed_videos}"
    )

    print(
        f"Skipped                   : "
        f"{skipped_videos}"
    )

    print(
        f"Failed                    : "
        f"{failed_videos}"
    )

    print()
    print("Expected dataset:")
    print("Train = 383")
    print("Val   = 79")
    print("Test  = 298")
    print("Total = 760")

    print()
    print("Expected output:")
    print("(number_of_frames, 126)")

    print()
    print("IMPORTANT:")
    print("Existing 30-frame files were overwritten")
    print("with full-length 126-feature sequences.")

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()