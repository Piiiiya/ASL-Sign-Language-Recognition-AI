import streamlit as st
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
import av
import threading

from pathlib import Path
from datetime import datetime
from collections import deque, Counter

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)


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

# Development:
# True  = application works at any time
# False = final submission works only 6 PM - 10 PM
DEVELOPER_MODE = True

# Minimum confidence required for a phrase word
PHRASE_CONFIDENCE_THRESHOLD = 0.80

# Number of stable predictions required before
# adding a sign to the phrase.
STABLE_PREDICTIONS_REQUIRED = 3

# Same sign cannot be added again immediately.
SAME_SIGN_COOLDOWN = 2.0


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# ============================================================
# PAGE
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

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD LABELS
# ============================================================

@st.cache_resource
def load_labels():

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

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = (
        mp.tasks.vision.HandLandmarker
    )

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(
                LANDMARKER_PATH
            )
        ),

        running_mode=RunningMode.IMAGE,

        num_hands=2,

        min_hand_detection_confidence=0.25,

        min_hand_presence_confidence=0.20,

        min_tracking_confidence=0.20,
    )

    return (
        HandLandmarker
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

            # ------------------------------------------------
            # Normalize relative to wrist
            # ------------------------------------------------

            wrist = points[0].copy()

            points = points - wrist

            # ------------------------------------------------
            # Scale normalization
            # ------------------------------------------------

            distances = np.linalg.norm(
                points,
                axis=1,
            )

            scale = np.max(distances)

            if scale > 1e-6:

                points = points / scale

            # ------------------------------------------------
            # Store according to handedness
            # ------------------------------------------------

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
        feature_vector.astype(
            np.float32
        ),
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
# REAL-TIME VIDEO PROCESSOR
# ============================================================

class ASLVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.model = load_model()

        self.id_to_label = load_labels()

        self.landmarker = create_landmarker()

        # ----------------------------------------------------
        # Sequence buffer
        # ----------------------------------------------------

        self.sequence_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_frames = 0
        self.active_frames = 0

        # ----------------------------------------------------
        # Current prediction
        # ----------------------------------------------------

        self.prediction = "Waiting"
        self.confidence = 0.0

        self.top_predictions = []

        # ----------------------------------------------------
        # Prediction timing
        # ----------------------------------------------------

        self.prediction_countdown = 0

        self.no_hand_frames = 0

        # ----------------------------------------------------
        # Stable prediction tracking
        # ----------------------------------------------------

        self.stable_label = None
        self.stable_count = 0

        # ----------------------------------------------------
        # Phrase
        # ----------------------------------------------------

        self.phrase = []

        self.last_added_sign = None
        self.last_added_time = 0.0

        # ----------------------------------------------------
        # Thread safety
        # ----------------------------------------------------

        self.lock = threading.Lock()

    # ========================================================
    # ADD SIGN TO PHRASE
    # ========================================================

    def add_to_phrase(
        self,
        label,
        confidence,
    ):

        if confidence < PHRASE_CONFIDENCE_THRESHOLD:
            return

        if label == "UNKNOWN":
            return

        current_time = datetime.now().timestamp()

        # ----------------------------------------------------
        # Stable prediction
        # ----------------------------------------------------

        if label == self.stable_label:

            self.stable_count += 1

        else:

            self.stable_label = label
            self.stable_count = 1

        # ----------------------------------------------------
        # Require multiple consecutive predictions
        # ----------------------------------------------------

        if (
            self.stable_count
            < STABLE_PREDICTIONS_REQUIRED
        ):

            return

        # ----------------------------------------------------
        # Prevent immediate duplicate
        # ----------------------------------------------------

        if (
            label == self.last_added_sign
            and
            current_time - self.last_added_time
            < SAME_SIGN_COOLDOWN
        ):

            return

        # ----------------------------------------------------
        # Add sign
        # ----------------------------------------------------

        self.phrase.append(label)

        self.last_added_sign = label

        self.last_added_time = current_time

        # Reset stability after adding
        self.stable_count = 0

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        self.total_frames += 1

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        try:

            result = self.landmarker.detect(
                mp_image
            )

        except Exception:

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24",
            )

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features, found_hand = (
            extract_features(result)
        )

        # ====================================================
        # HAND FOUND
        # ====================================================

        if found_hand:

            self.active_frames += 1

            self.no_hand_frames = 0

            self.sequence_buffer.append(
                features
            )

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            self.no_hand_frames += 1

        # ====================================================
        # PREDICTION
        # ====================================================

        if (
            len(self.sequence_buffer)
            >= SEQUENCE_LENGTH
        ):

            if self.prediction_countdown <= 0:

                sequence = np.array(
                    self.sequence_buffer,
                    dtype=np.float32,
                )

                sequence = resample_sequence(
                    sequence,
                    SEQUENCE_LENGTH,
                )

                X = np.expand_dims(
                    sequence,
                    axis=0,
                )

                try:

                    probabilities = (
                        self.model.predict(
                            X,
                            verbose=0,
                        )[0]
                    )

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

                    predicted_label = (
                        self.id_to_label.get(
                            predicted_id,
                            "UNKNOWN",
                        )
                    )

                    # ----------------------------------------
                    # Top 3
                    # ----------------------------------------

                    top_indices = (
                        np.argsort(
                            probabilities
                        )[::-1][:3]
                    )

                    top_predictions = []

                    for idx in top_indices:

                        top_predictions.append(
                            (
                                self.id_to_label.get(
                                    int(idx),
                                    "UNKNOWN",
                                ),
                                float(
                                    probabilities[idx]
                                ),
                            )
                        )

                    # ----------------------------------------
                    # Save prediction
                    # ----------------------------------------

                    with self.lock:

                        self.prediction = (
                            predicted_label
                        )

                        self.confidence = (
                            confidence
                        )

                        self.top_predictions = (
                            top_predictions
                        )

                        # ------------------------------------
                        # Phrase builder
                        # ------------------------------------

                        self.add_to_phrase(
                            predicted_label,
                            confidence,
                        )

                    # Predict periodically
                    self.prediction_countdown = 12

                except Exception:

                    pass

            else:

                self.prediction_countdown -= 1

        # ====================================================
        # CLEAR BUFFER AFTER HAND DISAPPEARS
        # ====================================================

        if self.no_hand_frames >= 15:

            self.sequence_buffer.clear()

            self.prediction_countdown = 0

            self.stable_label = None

            self.stable_count = 0

        # ====================================================
        # GET CURRENT STATE
        # ====================================================

        with self.lock:

            prediction = self.prediction

            confidence = self.confidence

            total_frames = self.total_frames

            active_frames = self.active_frames

            sequence_count = len(
                self.sequence_buffer
            )

            phrase_text = " ".join(
                self.phrase
            )

        # ====================================================
        # DRAW VIDEO UI
        # ====================================================

        # Header
        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1], 115),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            img,
            "ASL Sign Language AI",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Prediction
        cv2.putText(
            img,
            f"Sign: {prediction}",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        # Confidence
        cv2.putText(
            img,
            f"Confidence: {confidence * 100:.1f}%",
            (15, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # ----------------------------------------------------
        # Phrase on video
        # ----------------------------------------------------

        if phrase_text:

            display_phrase = phrase_text

            # Keep display from becoming too wide.
            if len(display_phrase) > 55:

                display_phrase = (
                    "..." + display_phrase[-52:]
                )

            cv2.putText(
                img,
                f"Phrase: {display_phrase}",
                (15, img.shape[0] - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        bottom_text = (
            f"Hand: "
            f"{'Detected' if found_hand else 'Not detected'}"
            f" | Active: {active_frames}"
            f" | Sequence: "
            f"{sequence_count}/{SEQUENCE_LENGTH}"
        )

        cv2.putText(
            img,
            bottom_text,
            (15, img.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # RETURN FRAME
        # ====================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )


# ============================================================
# VIDEO ANALYSIS
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

    # --------------------------------------------------------
    # Limit processing
    # --------------------------------------------------------

    max_frames = 120

    if total_frames > max_frames:

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            max_frames,
        ).astype(int)

        frame_indices = set(
            frame_indices.tolist()
        )

    else:

        frame_indices = None

    active_features = []

    processed_frames = 0
    active_frames = 0

    frame_number = 0

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

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

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

    cap.release()

    # --------------------------------------------------------
    # No hands
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
    # Top 3
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
# MAIN
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
        "**Features:** 126"
    )

    st.write(
        "**Phrase threshold:** 80%"
    )

    st.divider()

    st.header(
        "🤟 Supported Signs"
    )

    for i in range(
        len(id_to_label)
    ):

        st.write(
            f"{i + 1}. "
            f"{id_to_label[i]}"
        )


# ============================================================
# REAL-TIME WEBCAM
# ============================================================

st.subheader(
    "📷 Real-Time ASL Recognition"
)

st.info(
    "Click START and allow browser camera access."
)

ctx = webrtc_streamer(

    key="asl-live-camera",

    video_processor_factory=(
        ASLVideoProcessor
    ),

    rtc_configuration=(
        RTC_CONFIGURATION
    ),

    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    async_processing=True,
)


# ============================================================
# LIVE INFORMATION
# ============================================================

if ctx.video_processor:

    processor = ctx.video_processor

    with processor.lock:

        current_prediction = (
            processor.prediction
        )

        current_confidence = (
            processor.confidence
        )

        current_active = (
            processor.active_frames
        )

        current_sequence = len(
            processor.sequence_buffer
        )

        current_top = list(
            processor.top_predictions
        )

        current_phrase = list(
            processor.phrase
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Sign",
            current_prediction,
        )

    with col2:

        st.metric(
            "Confidence",
            f"{current_confidence * 100:.1f}%",
        )

    with col3:

        st.metric(
            "Active Frames",
            current_active,
        )

    with col4:

        st.metric(
            "Sequence",
            f"{current_sequence}/30",
        )

    # ========================================================
    # PHRASE
    # ========================================================

    st.divider()

    st.subheader(
        "📝 Recognized Phrase"
    )

    if current_phrase:

        st.markdown(
            "### "
            + " ".join(
                current_phrase
            )
        )

    else:

        st.info(
            "No signs added to the phrase yet."
        )

    # --------------------------------------------------------
    # Clear phrase
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Phrase",
        key="clear_phrase",
    ):

        with processor.lock:

            processor.phrase.clear()

            processor.last_added_sign = None

            processor.last_added_time = 0.0

            processor.stable_label = None

            processor.stable_count = 0

        st.rerun()

    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    if current_top:

        st.divider()

        st.subheader(
            "🔎 Latest Top Predictions"
        )

        for label, confidence in current_top:

            st.write(
                f"**{label}** — "
                f"{confidence * 100:.2f}%"
            )

            st.progress(
                confidence
            )


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.divider()

st.subheader(
    "📹 Upload ASL Sign Video"
)

uploaded_file = st.file_uploader(
    "Upload an MP4 video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
    ],
)


if uploaded_file is not None:

    temp_dir = PROJECT_DIR / "outputs"

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_video = (
        temp_dir
        / "uploaded_video.mp4"
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

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    if st.button(
        "🔎 Analyze Uploaded Video",
        type="primary",
        key="analyze_uploaded_video",
    ):

        with st.spinner(
            "Analyzing video..."
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

        # ----------------------------------------------------
        # Failed
        # ----------------------------------------------------

        if not result["success"]:

            st.error(
                result["message"]
            )

            st.warning(
                f"""
Frames: {result.get("total_frames", 0)}

Processed: {result.get("processed_frames", 0)}

Active hand frames: {result.get("active_frames", 0)}
"""
            )

        # ----------------------------------------------------
        # Successful
        # ----------------------------------------------------

        else:

            st.success(
                "✅ Prediction completed."
            )

            st.divider()

            st.markdown(
                f"""
# 🤟 {result["label"]}

### Confidence:

{result["confidence"] * 100:.2f}%
"""
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
            # Top 3
            # ------------------------------------------------

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
            # Diagnostics
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
                    "Very few frames contain "
                    "detected hands."
                )

            elif active_ratio < 25:

                st.info(
                    "Some hand frames were "
                    "detected, but detection "
                    "was intermittent."
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