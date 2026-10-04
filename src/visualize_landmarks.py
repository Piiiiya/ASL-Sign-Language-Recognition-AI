from pathlib import Path
import cv2
import mediapipe as mp

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = (
    PROJECT_DIR
    / "outputs"
    / "confusion_samples"
    / "BREAKFAST1"
    / "video_01"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "landmark_visualization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Find the original extracted frame images
frames = sorted(INPUT_VIDEO.glob("frame_*.jpg"))

if not frames:
    raise FileNotFoundError(
        f"No frames found in {INPUT_VIDEO}"
    )

# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

# ============================================================
# DRAW LANDMARKS
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    print("=" * 80)
    print("MEDIA PIPE LANDMARK VISUALIZATION")
    print("=" * 80)

    for frame_path in frames:

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read: {frame_path}")
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        hand_count = len(result.hand_landmarks)

        print(
            f"{frame_path.name}: "
            f"{hand_count} hand(s) detected"
        )

        # ----------------------------------------------------
        # Draw each detected hand
        # ----------------------------------------------------

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            # Draw connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            h, w = image.shape[:2]

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

                cv2.circle(
                    image,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            for a, b in connections:

                if a < len(points) and b < len(points):

                    cv2.line(
                        image,
                        points[a],
                        points[b],
                        (255, 0, 0),
                        2
                    )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = f"Hand {hand_index + 1}"

            cv2.putText(
                image,
                label,
                (
                    points[0][0],
                    max(25, points[0][1] - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ====================================================
        # SAVE
        # ====================================================

        output_path = (
            OUTPUT_DIR
            / frame_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print("\nOutput folder:")
print(OUTPUT_DIR)