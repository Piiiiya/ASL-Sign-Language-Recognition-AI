
import streamlit as st

st.set_page_config(
    page_title="ASL Minimal Test",
    page_icon="🧪",
)

st.title("ASL Deployment - Minimal Test")

st.success("Streamlit is running successfully!")

st.write("This app does not import TensorFlow, MediaPipe, OpenCV, or PyAV.")

st.info("If you can see this message, the basic Streamlit app is working.")
