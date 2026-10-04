from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "confusion_samples"
    / "BREAKFAST1"
    / "video_01"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

frames = sorted(INPUT_DIR.glob("frame_*.jpg"))

print("=" * 80)
print("LANDMARK QUALITY CHECK")
print("=" * 80)

with HandLandmarker.create_from_options(options) as landmarker:

    for frame_path in frames:

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"{frame_path.name}: could not read")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        print(f"\n{frame_path.name}")
        print("-" * 50)

        hand_count = len(result.hand_landmarks)

        print(f"Hands detected: {hand_count}")

        if hand_count == 0:
            print("No landmarks detected.")
            continue

        for i, landmarks in enumerate(result.hand_landmarks):

            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32
            )

            xs = points[:, 0]
            ys = points[:, 1]
            zs = points[:, 2]

            print(f"\nHand {i + 1}")

            if result.handedness:
                if i < len(result.handedness):
                    category = result.handedness[i][0]

                    print(
                        f"Handedness: "
                        f"{category.category_name}"
                    )

                    print(
                        f"Hand confidence: "
                        f"{category.score:.4f}"
                    )

            print(
                f"X range: "
                f"{xs.min():.3f} → {xs.max():.3f}"
            )

            print(
                f"Y range: "
                f"{ys.min():.3f} → {ys.max():.3f}"
            )

            print(
                f"Z range: "
                f"{zs.min():.3f} → {zs.max():.3f}"
            )

            print(
                f"Wrist (landmark 0): "
                f"X={points[0,0]:.3f}, "
                f"Y={points[0,1]:.3f}, "
                f"Z={points[0,2]:.3f}"
            )

            print(
                f"Thumb tip (4): "
                f"X={points[4,0]:.3f}, "
                f"Y={points[4,1]:.3f}"
            )

            print(
                f"Index tip (8): "
                f"X={points[8,0]:.3f}, "
                f"Y={points[8,1]:.3f}"
            )

            print(
                f"Middle tip (12): "
                f"X={points[12,0]:.3f}, "
                f"Y={points[12,1]:.3f}"
            )

            print(
                f"Ring tip (16): "
                f"X={points[16,0]:.3f}, "
                f"Y={points[16,1]:.3f}"
            )

            print(
                f"Pinky tip (20): "
                f"X={points[20,0]:.3f}, "
                f"Y={points[20,1]:.3f}"
            )

            # Check whether landmarks are inside the image
            inside = np.all(
                (xs >= 0) &
                (xs <= 1) &
                (ys >= 0) &
                (ys <= 1)
            )

            print(
                f"Landmarks inside image: "
                f"{inside}"
            )

            # Check whether landmarks collapse to one point
            x_spread = xs.max() - xs.min()
            y_spread = ys.max() - ys.min()

            print(
                f"Hand width (normalized): "
                f"{x_spread:.3f}"
            )

            print(
                f"Hand height (normalized): "
                f"{y_spread:.3f}"
            )

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)