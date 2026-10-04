import streamlit as st
import subprocess
import sys

st.set_page_config(
    page_title="ASL Import Isolation Test",
    layout="centered"
)

st.title("ASL Import Isolation Test")

st.write(
    "Testing Python libraries in separate processes "
    "to identify the segmentation fault."
)

tests = {
    "NumPy": "import numpy; print('NumPy:', numpy.__version__)",
    "OpenCV": "import cv2; print('OpenCV:', cv2.__version__)",
    "MediaPipe": "import mediapipe; print('MediaPipe:', mediapipe.__version__)",
    "TensorFlow": "import tensorflow as tf; print('TensorFlow:', tf.__version__)",
    "PyAV": "import av; print('PyAV:', av.__version__)",
    "WebRTC": (
        "from streamlit_webrtc import "
        "webrtc_streamer, VideoProcessorBase, RTCConfiguration; "
        "print('WebRTC import successful')"
    ),
}

for name, code in tests.items():

    st.subheader(f"Testing {name}")

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            st.success(f"{name}: PASSED")

            if result.stdout:
                st.code(result.stdout.strip())

        else:
            st.error(
                f"{name}: FAILED "
                f"(exit code {result.returncode})"
            )

            if result.stderr:
                st.code(result.stderr[-3000:])

    except subprocess.TimeoutExpired:
        st.error(f"{name}: Timed out after 120 seconds")

    except Exception as error:
        st.error(f"{name}: {error}")

st.success("Isolation test completed.")