
import streamlit as st

st.set_page_config(page_title="MediaPipe Deployment Test")

st.title("MediaPipe Deployment Test")

st.write("Starting MediaPipe import...")

import mediapipe as mp

st.success("MediaPipe imported successfully!")

st.write("MediaPipe version:", mp.__version__)

st.write("Checking MediaPipe Tasks API...")

st.write("Tasks available:", hasattr(mp, "tasks"))

if hasattr(mp, "tasks"):
    st.success("MediaPipe Tasks API is available!")
else:
    st.error("MediaPipe Tasks API is not available.")

st.success("MediaPipe test completed successfully!")
