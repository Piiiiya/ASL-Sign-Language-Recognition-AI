
import streamlit as st

st.set_page_config(page_title="TensorFlow Deployment Test")

st.title("TensorFlow Deployment Test")

st.write("Starting TensorFlow import...")

import tensorflow as tf

st.success("TensorFlow imported successfully!")

st.write("TensorFlow version:", tf.__version__)

st.write("Checking CPU availability...")

devices = tf.config.list_physical_devices("CPU")

st.write("CPU devices:", devices)

st.success("TensorFlow test completed successfully!")
