from pathlib import Path
from collections import deque
import json

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "asl_lstm_active_best.keras"
)

LABEL_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences_active"
    / "label_mapping.json"
)

HAND_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "hand_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
NUM_FEATURES = 126

CONFIDENCE_THRESHOLD = 0.50

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not MODEL_PATH.exists():
    print("ERROR: ASL model not found:")
    print(MODEL_PATH)
    raise SystemExit

if not LABEL_PATH.exists():
    print("ERROR: Label mapping not found:")
    print(LABEL_PATH)
    raise SystemExit

if not HAND_MODEL_PATH.exists():
    print("ERROR: MediaPipe hand model not found:")
    print(HAND_MODEL_PATH)
    raise SystemExit

print("ASL model :", MODEL_PATH)
print("Labels    :", LABEL_PATH)
print("Hand model:", HAND_MODEL_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING ASL MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# VERIFY MODEL INPUT
# ============================================================

expected_shape = (
    None,
    SEQUENCE_LENGTH,
    NUM_FEATURES
)

if model.input_shape != expected_shape:

    print()
    print("WARNING: Model input shape is different.")
    print("Expected:", expected_shape)
    print("Actual  :", model.input_shape)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8"
) as f:

    label_mapping = json.load(f)


# Your JSON contains:
#
# {
#     "label_to_id": {...},
#     "id_to_label": {...}
# }

id_to_label = {
    int(key): value
    for key, value
    in label_mapping["id_to_label"].items()
}


print()
print("Labels:")

for class_id in sorted(id_to_label):

    print(
        f"{class_id}: "
        f"{id_to_label[class_id]}"
    )


# ============================================================
# MEDIA PIPE HAND LANDMARKER
# ============================================================

mp_vision = mp.tasks.vision

base_options = mp.tasks.BaseOptions(
    model_asset_path=str(HAND_MODEL_PATH)
)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=2
)

landmarker = (
    mp_vision.HandLandmarker
    .create_from_options(options)
)


# ============================================================
# EXTRACT LANDMARKS
# ============================================================

def extract_landmarks(result):

    left_hand = np.zeros(
        63,
        dtype=np.float32
    )

    right_hand = np.zeros(
        63,
        dtype=np.float32
    )

    if not result.hand_landmarks:

        return np.concatenate(
            [
                left_hand,
                right_hand
            ]
        )

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        landmarks = []

        for landmark in hand_landmarks:

            landmarks.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
            )

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Identify Left / Right hand
        # ----------------------------------------------------

        if (
            result.handedness
            and
            hand_index < len(
                result.handedness
            )
            and
            len(
                result.handedness[
                    hand_index
                ]
            ) > 0
        ):

            handedness = (
                result.handedness[
                    hand_index
                ][0].category_name
            )

            if handedness == "Left":

                left_hand = landmarks

            elif handedness == "Right":

                right_hand = landmarks

    return np.concatenate(
        [
            left_hand,
            right_hand
        ]
    )


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_frame(features):

    features = features.astype(
        np.float32
    )

    normalized = np.zeros(
        126,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # LEFT HAND
    # --------------------------------------------------------

    left = features[:63].reshape(
        21,
        3
    )

    if not np.allclose(
        left,
        0
    ):

        wrist = left[0].copy()

        left = left - wrist

        distances = np.linalg.norm(
            left,
            axis=1
        )

        scale = np.max(
            distances
        )

        if scale > 1e-6:

            left = left / scale

        normalized[:63] = (
            left.flatten()
        )

    # --------------------------------------------------------
    # RIGHT HAND
    # --------------------------------------------------------

    right = features[63:].reshape(
        21,
        3
    )

    if not np.allclose(
        right,
        0
    ):

        wrist = right[0].copy()

        right = right - wrist

        distances = np.linalg.norm(
            right,
            axis=1
        )

        scale = np.max(
            distances
        )

        if scale > 1e-6:

            right = right / scale

        normalized[63:] = (
            right.flatten()
        )

    return normalized


# ============================================================
# START WEBCAM
# ============================================================

print()
print("=" * 70)
print("STARTING WEBCAM")
print("=" * 70)

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    landmarker.close()

    raise SystemExit


# ============================================================
# REQUEST CAMERA FORMAT
# ============================================================

print()
print("Requesting camera format...")

# MJPG is commonly more reliable with Windows webcams.
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        "M",
        "J",
        "P",
        "G"
    )
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)


# ============================================================
# READ ACTUAL CAMERA FORMAT
# ============================================================

actual_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

actual_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

actual_fps = cap.get(
    cv2.CAP_PROP_FPS
)


print(
    f"Requested resolution: "
    f"{CAMERA_WIDTH} x {CAMERA_HEIGHT}"
)

print(
    f"Actual resolution   : "
    f"{actual_width} x {actual_height}"
)

print(
    f"Actual FPS          : "
    f"{actual_fps:.1f}"
)


if (
    actual_width < 320
    or
    actual_height < 240
):

    print()
    print(
        "WARNING: Camera returned "
        "a very low resolution."
    )

    print(
        "Hand detection may not work correctly."
    )

    print(
        f"Current resolution: "
        f"{actual_width} x {actual_height}"
    )


# ============================================================
# SEQUENCE BUFFER
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# DISPLAY VARIABLES
# ============================================================

prediction = "Waiting..."

confidence = 0.0

frame_counter = 0

active_frame_counter = 0

hand_detection_counter = 0


print()
print("Webcam opened successfully.")
print()
print("Instructions:")
print("1. Sit in front of the camera.")
print("2. Keep your hand clearly visible.")
print("3. Keep the complete hand inside the frame.")
print("4. Show one of the trained ASL signs.")
print("5. Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print()
            print(
                "ERROR: Could not read webcam frame."
            )

            break

        frame_counter += 1

        # ----------------------------------------------------
        # Frame diagnostics
        # ----------------------------------------------------

        if frame_counter % 30 == 0:

            print(
                f"Frame {frame_counter}: "
                f"{frame.shape}"
            )

        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )

        # ----------------------------------------------------
        # Count detected hands
        # ----------------------------------------------------

        detected_hands = 0

        if result.hand_landmarks:

            detected_hands = len(
                result.hand_landmarks
            )

            hand_detection_counter += 1

        # ----------------------------------------------------
        # Extract landmarks
        # ----------------------------------------------------

        features = extract_landmarks(
            result
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        features = normalize_frame(
            features
        )

        # ----------------------------------------------------
        # Check active frame
        # ----------------------------------------------------

        has_hand = not np.allclose(
            features,
            0
        )

        if has_hand:

            sequence.append(
                features
            )

            active_frame_counter += 1

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]

            predicted_id = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = float(
                probabilities[
                    predicted_id
                ]
            )

            predicted_label = id_to_label.get(
                predicted_id,
                "Unknown"
            )

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction = predicted_label

            else:

                prediction = "Uncertain"

        else:

            prediction = "Collecting..."


        # ====================================================
        # TOP DISPLAY PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (
                frame.shape[1],
                145
            ),
            (0, 0, 0),
            -1
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Sign: {prediction}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Confidence: "
            f"{confidence * 100:.1f}%",
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frames: "
            f"{len(sequence)}/"
            f"{SEQUENCE_LENGTH}",
            (15, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Hands detected
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Hands: {detected_hands}",
            (15, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Quit instruction
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Q = Quit",
            (
                frame.shape[1] - 110,
                frame.shape[0] - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DRAW HAND LANDMARKS
        # ====================================================

        if result.hand_landmarks:

            for hand_landmarks in (
                result.hand_landmarks
            ):

                for landmark in hand_landmarks:

                    x = int(
                        landmark.x
                        * frame.shape[1]
                    )

                    y = int(
                        landmark.y
                        * frame.shape[0]
                    )

                    if (
                        0 <= x <
                        frame.shape[1]
                        and
                        0 <= y <
                        frame.shape[0]
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "ASL Sign Language Detection",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key == ord("q"):

            print()
            print(
                "Q pressed. "
                "Stopping webcam..."
            )

            break

        if key == 27:

            print()
            print(
                "ESC pressed. "
                "Stopping webcam..."
            )

            break


# ============================================================
# CTRL+C
# ============================================================

except KeyboardInterrupt:

    print()
    print(
        "Stopped by user."
    )


# ============================================================
# CLEANUP
# ============================================================

finally:

    cap.release()

    cv2.destroyAllWindows()

    landmarker.close()

    print()
    print("=" * 70)
    print("WEBCAM STOPPED")
    print("=" * 70)

    print(
        f"Total frames captured : "
        f"{frame_counter}"
    )

    print(
        f"Active hand frames    : "
        f"{active_frame_counter}"
    )

    print(
        f"Frames with hand      : "
        f"{hand_detection_counter}"
    )