import streamlit as st
import subprocess
import sys

st.set_page_config(
    page_title="ASL Library Combination Test",
    layout="centered"
)

st.title("ASL Library Combination Test")

tests = {
    "OpenCV + MediaPipe": """
import cv2
print("OpenCV imported")
import mediapipe
print("MediaPipe imported")
""",

    "MediaPipe + OpenCV": """
import mediapipe
print("MediaPipe imported")
import cv2
print("OpenCV imported")
""",

    "TensorFlow + MediaPipe": """
import tensorflow as tf
print("TensorFlow imported")
import mediapipe
print("MediaPipe imported")
""",

    "MediaPipe + TensorFlow": """
import mediapipe
print("MediaPipe imported")
import tensorflow as tf
print("TensorFlow imported")
""",

    "OpenCV + TensorFlow": """
import cv2
print("OpenCV imported")
import tensorflow as tf
print("TensorFlow imported")
""",

    "TensorFlow + OpenCV": """
import tensorflow as tf
print("TensorFlow imported")
import cv2
print("OpenCV imported")
""",

    "OpenCV + MediaPipe + TensorFlow": """
import cv2
print("OpenCV imported")
import mediapipe
print("MediaPipe imported")
import tensorflow as tf
print("TensorFlow imported")
""",

    "TensorFlow + MediaPipe + OpenCV": """
import tensorflow as tf
print("TensorFlow imported")
import mediapipe
print("MediaPipe imported")
import cv2
print("OpenCV imported")
""",

    "All libraries": """
import numpy
print("NumPy imported")

import cv2
print("OpenCV imported")

import mediapipe
print("MediaPipe imported")

import tensorflow as tf
print("TensorFlow imported")

import av
print("PyAV imported")

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)
print("WebRTC imported")
"""
}

for name, code in tests.items():

    st.subheader(name)

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            st.success("PASSED")

        else:
            st.error(
                f"FAILED — exit code: {result.returncode}"
            )

        if result.stdout:
            st.code(result.stdout.strip())

        if result.stderr:
            st.text("Error output:")
            st.code(result.stderr[-2000:])

    except subprocess.TimeoutExpired:
        st.error("Test timed out after 120 seconds")

    except Exception as error:
        st.error(str(error))

st.success("All combination tests completed.")