
import streamlit as st

st.set_page_config(page_title="Combined Library Test")
st.title("ASL Combined Library Test")

st.write("Testing all libraries in one Python process...")

try:
    st.write("1. Importing NumPy...")
    import numpy as np
    st.success(f"NumPy {np.__version__} imported")

    st.write("2. Importing OpenCV...")
    import cv2
    st.success(f"OpenCV {cv2.__version__} imported")

    st.write("3. Importing MediaPipe...")
    import mediapipe as mp
    st.success(f"MediaPipe {mp.__version__} imported")

    st.write("4. Importing TensorFlow...")
    import tensorflow as tf
    st.success(f"TensorFlow {tf.__version__} imported")

    st.write("5. Importing PyAV...")
    import av
    st.success(f"PyAV {av.__version__} imported")

    st.write("6. Importing Streamlit WebRTC...")
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
    )
    st.success("Streamlit WebRTC imported")

    st.success("ALL LIBRARIES IMPORTED SUCCESSFULLY!")

except Exception as error:
    st.error(f"Import failed: {error}")
    st.exception(error)
