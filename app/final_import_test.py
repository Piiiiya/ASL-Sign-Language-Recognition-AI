import streamlit as st
import subprocess
import sys

st.set_page_config(
    page_title="ASL Final Import Test",
    layout="centered"
)

st.title("ASL Final Import Test")

tests = {
    "1. NumPy + OpenCV + MediaPipe + TensorFlow": """
import numpy
print("NumPy OK", flush=True)

import cv2
print("OpenCV OK", flush=True)

import mediapipe
print("MediaPipe OK", flush=True)

import tensorflow as tf
print("TensorFlow OK", flush=True)
""",

    "2. Previous libraries + PyAV": """
import numpy
print("NumPy OK", flush=True)

import cv2
print("OpenCV OK", flush=True)

import mediapipe
print("MediaPipe OK", flush=True)

import tensorflow as tf
print("TensorFlow OK", flush=True)

import av
print("PyAV OK", flush=True)
""",

    "3. Previous libraries + WebRTC": """
import numpy
print("NumPy OK", flush=True)

import cv2
print("OpenCV OK", flush=True)

import mediapipe
print("MediaPipe OK", flush=True)

import tensorflow as tf
print("TensorFlow OK", flush=True)

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

print("WebRTC OK", flush=True)
""",

    "4. PyAV + WebRTC": """
import av
print("PyAV OK", flush=True)

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

print("WebRTC OK", flush=True)
""",

    "5. All libraries in original order": """
import numpy
print("NumPy OK", flush=True)

import cv2
print("OpenCV OK", flush=True)

import mediapipe
print("MediaPipe OK", flush=True)

import tensorflow as tf
print("TensorFlow OK", flush=True)

import av
print("PyAV OK", flush=True)

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

print("WebRTC OK", flush=True)
""",

    "6. All libraries, WebRTC before TensorFlow": """
import numpy
print("NumPy OK", flush=True)

import cv2
print("OpenCV OK", flush=True)

import mediapipe
print("MediaPipe OK", flush=True)

import av
print("PyAV OK", flush=True)

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

print("WebRTC OK", flush=True)

import tensorflow as tf
print("TensorFlow OK", flush=True)
"""
}

for name, code in tests.items():

    st.subheader(name)

    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
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

st.success("Final import test completed.")