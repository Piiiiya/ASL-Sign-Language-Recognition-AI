
import streamlit as st

st.set_page_config(page_title="ASL Deployment Diagnostic")
st.title("ASL Deployment Diagnostic")
st.write("Checking libraries one by one...")

def test_streamlit():
    import streamlit

def test_numpy():
    import numpy

def test_cv2():
    import cv2

def test_mediapipe():
    import mediapipe

def test_tensorflow():
    import tensorflow

def test_av():
    import av

def test_webrtc():
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
    )

checks = [
    ("Streamlit", test_streamlit),
    ("NumPy", test_numpy),
    ("OpenCV", test_cv2),
    ("MediaPipe", test_mediapipe),
    ("TensorFlow", test_tensorflow),
    ("PyAV", test_av),
    ("Streamlit WebRTC", test_webrtc),
]

for name, function in checks:
    try:
        st.write(f"Testing: {name}")
        function()
        st.success(f"{name}: Import successful")
    except Exception as error:
        st.error(f"{name} failed: {error}")
        st.stop()

st.success("All library imports passed!")
