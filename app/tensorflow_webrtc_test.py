import streamlit as st

st.set_page_config(page_title="TensorFlow + WebRTC Test")

st.title("TensorFlow + WebRTC Compatibility Test")

st.write("Testing TensorFlow and WebRTC in the same process.")

try:
    st.write("Step 1: Importing TensorFlow...")
    import tensorflow as tf

    st.success(f"TensorFlow imported: {tf.__version__}")

    st.write("Step 2: Importing WebRTC...")
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
    )

    st.success("WebRTC imported successfully.")

    st.write("Step 3: Checking TensorFlow CPU...")
    devices = tf.config.list_physical_devices("CPU")

    st.write("CPU devices:", devices)

    st.success("Both libraries passed the test!")

except Exception as error:
    st.error(f"Python exception: {error}")
    st.exception(error)