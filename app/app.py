
import streamlit as st
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf

from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "asl_lstm_active_best.keras"
)

LANDMARKER_PATH = (
    PROJECT_DIR
    / "models"
    / "hand_landmarker.task"
)

LABEL_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sequences_active"
    / "label_mapping.json"
)

SEQUENCE_LENGTH = 30
FEATURES = 126

# True  = application works at any time
# False = final submission works only 6 PM - 10 PM
DEVELOPER_MODE = True

MAX_VIDEO_FRAMES = 120


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ASL Sign Language AI",
    page_icon="🤟",
    layout="wide",
)


# ============================================================
# OPERATIONAL TIME
# ============================================================

def is_operational_time():

    if DEVELOPER_MODE:
        return True

    current_time = datetime.now().time()

    start_time = datetime.strptime(
        "18:00",
        "%H:%M",
    ).time()

    end_time = datetime.strptime(
        "22:00",
        "%H:%M",
    ).time()

    return start_time <= current_time < end_time


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD LABELS
# ============================================================

@st.cache_resource
def load_labels():

    if not LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found: {LABEL_PATH}"
        )

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return {
        int(k): v
        for k, v in data["id_to_label"].items()
    }


# ============================================================
# CREATE MEDIAPIPE LANDMARKER
# ============================================================

def create_landmarker():

    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe task file not found: {LANDMARKER_PATH}"
        )

    options = mp.tasks.vision.HandLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(
                LANDMARKER_PATH
            )
        ),

        running_mode=mp.tasks.vision.RunningMode.IMAGE,

        num_hands=2,

        min_hand_detection_confidence=0.25,

        min_hand_presence_confidence=0.20,

        min_tracking_confidence=0.20,
    )

    return (
        mp.tasks.vision.HandLandmarker
        .create_from_options(options)
    )


# ============================================================
# EXTRACT HAND FEATURES
# ============================================================

def extract_features(result):

    left = np.zeros(
        (21, 3),
        dtype=np.float32,
    )

    right = np.zeros(
        (21, 3),
        dtype=np.float32,
    )

    found_hand = False

    if (
        result.hand_landmarks
        and result.handedness
    ):

        for hand_landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness,
        ):

            if not handedness:
                continue

            hand_type = (
                handedness[0].category_name
            )

            points = np.array(
                [
                    [
                        lm.x,
                        lm.y,
                        lm.z,
                    ]
                    for lm in hand_landmarks
                ],
                dtype=np.float32,
            )

            if points.shape != (21, 3):
                continue

            # Normalize relative to wrist
            wrist = points[0].copy()

            points = points - wrist

            # Scale normalization
            distances = np.linalg.norm(
                points,
                axis=1,
            )

            scale = np.max(distances)

            if scale > 1e-6:
                points = points / scale

            # Store according to handedness
            if hand_type == "Left":

                left = points
                found_hand = True

            elif hand_type == "Right":

                right = points
                found_hand = True

    feature_vector = np.concatenate(
        [
            left.flatten(),
            right.flatten(),
        ]
    )

    return (
        feature_vector.astype(np.float32),
        found_hand,
    )


# ============================================================
# RESAMPLE SEQUENCE
# ============================================================

def resample_sequence(
    sequence,
    target_length=30,
):

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if len(sequence) == 0:
        return None

    if len(sequence) == target_length:
        return sequence

    old_indices = np.linspace(
        0,
        len(sequence) - 1,
        len(sequence),
    )

    new_indices = np.linspace(
        0,
        len(sequence) - 1,
        target_length,
    )

    output = np.zeros(
        (
            target_length,
            sequence.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(
        sequence.shape[1]
    ):

        output[:, feature_index] = np.interp(
            new_indices,
            old_indices,
            sequence[:, feature_index],
        )

    return output


# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(
    video_path,
    landmarker,
    model,
    id_to_label,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return {
            "success": False,
            "message": "Could not open video.",
        }

    try:

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30

        # ----------------------------------------------------
        # Limit processing to a maximum of 120 frames
        # ----------------------------------------------------

        if total_frames > MAX_VIDEO_FRAMES:

            frame_indices = set(
                np.linspace(
                    0,
                    total_frames - 1,
                    MAX_VIDEO_FRAMES,
                ).astype(int).tolist()
            )

        else:

            frame_indices = None

        active_features = []

        processed_frames = 0
        active_frames = 0

        frame_number = 0

        # ----------------------------------------------------
        # Process video frames
        # ----------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if (
                frame_indices is not None
                and frame_number not in frame_indices
            ):

                frame_number += 1
                continue

            processed_frames += 1

            # BGR to RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            # MediaPipe detection
            result = landmarker.detect(
                mp_image
            )

            features, found_hand = (
                extract_features(result)
            )

            if found_hand:

                active_features.append(
                    features
                )

                active_frames += 1

            frame_number += 1

    finally:

        cap.release()

    # --------------------------------------------------------
    # No hands detected
    # --------------------------------------------------------

    if active_frames == 0:

        return {
            "success": False,
            "message": (
                "No hand landmarks detected. "
                "Try a video where the hand is clearly visible."
            ),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "active_frames": 0,
            "fps": fps,
        }

    # --------------------------------------------------------
    # Create sequence
    # --------------------------------------------------------

    sequence = resample_sequence(
        active_features,
        SEQUENCE_LENGTH,
    )

    if sequence is None:

        return {
            "success": False,
            "message": "Could not create sequence.",
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "active_frames": active_frames,
            "fps": fps,
        }

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    X = np.expand_dims(
        sequence,
        axis=0,
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict(
        X,
        verbose=0,
    )[0]

    predicted_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_id]
    )

    predicted_label = id_to_label.get(
        predicted_id,
        "UNKNOWN",
    )

    # --------------------------------------------------------
    # Top 3 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = []

    for idx in top_indices:

        top_predictions.append(
            {
                "label": id_to_label.get(
                    int(idx),
                    "UNKNOWN",
                ),
                "confidence": float(
                    probabilities[idx]
                ),
            }
        )

    return {
        "success": True,
        "label": predicted_label,
        "confidence": confidence,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "active_frames": active_frames,
        "fps": fps,
        "top_predictions": top_predictions,
    }


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title(
    "🤟 ASL Sign Language AI"
)

st.caption(
    "Computer Vision + MediaPipe + BiLSTM"
)


# ============================================================
# TIME CHECK
# ============================================================

if not is_operational_time():

    current = datetime.now().strftime(
        "%I:%M %p"
    )

    st.warning(
        "The ASL Sign Language AI system "
        "is operational only from "
        "6:00 PM to 10:00 PM."
    )

    st.info(
        f"Current time: {current}"
    )

    st.stop()


# ============================================================
# LOAD COMPONENTS
# ============================================================

try:

    model = load_model()

    id_to_label = load_labels()

except Exception as e:

    st.error(
        "Failed to load AI components."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Information"
    )

    st.write("**Model:**")

    st.code(
        MODEL_PATH.name
    )

    st.write(
        f"**Classes:** {len(id_to_label)}"
    )

    st.write(
        "**Input:** 30 active frames"
    )

    st.write(
        f"**Features:** {FEATURES}"
    )

    st.write(
        "**Recognition:** Uploaded video"
    )

    st.divider()

    st.header(
        "🤟 Supported Signs"
    )

    for i in range(
        len(id_to_label)
    ):

        st.write(
            f"{i + 1}. {id_to_label[i]}"
        )


# ============================================================
# APPLICATION INFORMATION
# ============================================================

st.subheader(
    "📹 ASL Sign Recognition"
)

st.info(
    "Upload a video showing a clearly visible ASL sign. "
    "The system will extract hand landmarks and predict "
    "the most likely sign."
)

st.caption(
    "Live webcam recognition is temporarily unavailable "
    "in this deployment while compatibility is being tested."
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.divider()

st.subheader(
    "📹 Upload ASL Sign Video"
)

uploaded_file = st.file_uploader(
    "Upload an ASL video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
    key="asl_video_upload",
)


if uploaded_file is not None:

    # --------------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------------

    temp_dir = (
        PROJECT_DIR
        / "outputs"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_extension = (
        Path(uploaded_file.name).suffix.lower()
    )

    if not file_extension:
        file_extension = ".mp4"

    temp_video = (
        temp_dir
        / f"uploaded_video{file_extension}"
    )

    with open(
        temp_video,
        "wb",
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # --------------------------------------------------------
    # Video preview
    # --------------------------------------------------------

    st.video(
        uploaded_file
    )

    st.write(
        f"**File:** {uploaded_file.name}"
    )

    st.write(
        f"**Size:** {uploaded_file.size / (1024 * 1024):.2f} MB"
    )

    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

    if st.button(
        "🔎 Analyze Uploaded Video",
        type="primary",
        key="analyze_uploaded_video",
    ):

        upload_landmarker = None

        try:

            with st.spinner(
                "Analyzing video and extracting hand landmarks..."
            ):

                upload_landmarker = (
                    create_landmarker()
                )

                result = analyze_video(
                    temp_video,
                    upload_landmarker,
                    model,
                    id_to_label,
                )

        except Exception as e:

            st.error(
                "An error occurred while analyzing the video."
            )

            st.exception(e)

            result = None

        finally:

            if upload_landmarker is not None:

                upload_landmarker.close()

        # ----------------------------------------------------
        # Failed analysis
        # ----------------------------------------------------

        if result is not None and not result["success"]:

            st.error(
                result["message"]
            )

            st.warning(
                f"""
Total frames: {result.get("total_frames", 0)}

Processed frames: {result.get("processed_frames", 0)}

Active hand frames: {result.get("active_frames", 0)}
"""
            )

        # ----------------------------------------------------
        # Successful prediction
        # ----------------------------------------------------

        elif result is not None:

            st.success(
                "✅ Prediction completed."
            )

            st.divider()

            st.markdown(
                f"""
# 🤟 {result["label"]}

### Model confidence: {result["confidence"] * 100:.2f}%
"""
            )

            st.caption(
                "Confidence is the model's predicted probability "
                "for this input, not a guarantee of correctness."
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Frames",
                    result["total_frames"],
                )

            with col2:

                st.metric(
                    "Processed Frames",
                    result["processed_frames"],
                )

            with col3:

                st.metric(
                    "Active Hand Frames",
                    result["active_frames"],
                )

            # ------------------------------------------------
            # Top 3 predictions
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔎 Top Predictions"
            )

            for prediction in result[
                "top_predictions"
            ]:

                label = prediction[
                    "label"
                ]

                confidence = prediction[
                    "confidence"
                ]

                st.write(
                    f"**{label}** — "
                    f"{confidence * 100:.2f}%"
                )

                st.progress(
                    confidence
                )

            # ------------------------------------------------
            # Detection diagnostics
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🧪 Detection Diagnostics"
            )

            active_ratio = (
                result["active_frames"]
                / max(
                    result["processed_frames"],
                    1,
                )
            ) * 100

            st.write(
                "Hand-active frame ratio: "
                f"**{active_ratio:.2f}%**"
            )

            if active_ratio < 10:

                st.warning(
                    "Very few frames contain detected hands. "
                    "Try improving the lighting and keeping "
                    "your hands clearly visible."
                )

            elif active_ratio < 25:

                st.info(
                    "Some hand frames were detected, but "
                    "detection was intermittent."
                )

            else:

                st.success(
                    "Good hand-frame detection."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤟 ASL Sign Language AI | "
    "Computer Vision + MediaPipe + BiLSTM"
)
