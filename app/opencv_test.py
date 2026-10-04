
import streamlit as st

st.set_page_config(page_title="OpenCV Deployment Test")

st.title("OpenCV Deployment Test")

st.write("Starting OpenCV import...")

import cv2

st.success("OpenCV imported successfully!")

st.write("OpenCV version:", cv2.__version__)

st.write("Checking image processing...")

import numpy as np

test_image = np.zeros((100, 100, 3), dtype=np.uint8)

st.write("Test image shape:", test_image.shape)

st.success("OpenCV image processing test completed!")
