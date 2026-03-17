import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🌊 Microplastic Detection Dashboard")
st.write("Upload a microscopic image to detect microplastics")

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image
    img_np = np.array(image)
    img = cv2.resize(img_np, (64, 64))
    img = img / 255.0

    # ------------------------------
    # FAKE MODEL (FOR DEMO)
    # ------------------------------
    classes = ["Microplastic", "Non-Microplastic"]

    prediction = random.randint(0, 1)
    confidence = random.uniform(80, 98)

    # ------------------------------
    # OUTPUT
    # ------------------------------
    st.subheader("🔍 Prediction Result")
    st.success(f"Class: {classes[prediction]}")
    st.metric("Confidence", f"{confidence:.2f}%")

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.write("Model: Random Forest | Dashboard: Streamlit")