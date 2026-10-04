import streamlit as st

st.set_page_config(page_title="OpenCV + MediaPipe Test")

st.title("OpenCV + MediaPipe Compatibility Test")

st.write("Starting compatibility test...")

try:
    st.write("1. Importing OpenCV...")
    import cv2
    st.success(f"OpenCV imported: {cv2.__version__}")

    st.write("2. Importing MediaPipe...")
    import mediapipe as mp
    st.success(f"MediaPipe imported: {mp.__version__}")

    st.write("3. Checking MediaPipe Tasks API...")
    st.success(f"Tasks API available: {hasattr(mp, 'tasks')}")

    st.success("OpenCV + MediaPipe test completed successfully!")

except Exception as e:
    st.error(f"Test failed: {e}")