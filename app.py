import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🌊 Microplastic Detection Dashboard")
st.write("Upload a microscopic image to detect microplastics")

# ------------------------------
# LOAD MODEL SAFELY
# ------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("rf_model.pkl"):
        model = joblib.load("rf_model.pkl")
        return model
    else:
        return None

rf_model = load_model()

# ------------------------------
# PREPROCESS IMAGE
# ------------------------------
def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten().reshape(1, -1)
    return img

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)
    processed = preprocess(img_np)

    if rf_model is not None:
        # Prediction
        prediction = rf_model.predict(processed)[0]
        probabilities = rf_model.predict_proba(processed)

        confidence = np.max(probabilities) * 100
        classes = ["Microplastic", "Non-Microplastic"]

        st.subheader("🔍 Prediction Result")
        st.success(f"Class: {classes[int(prediction)]}")
        st.metric("Confidence", f"{confidence:.2f}%")

    else:
        st.error("⚠️ Model file not found! Please upload rf_model.pkl to GitHub.")

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.write("Model: Random Forest | Dashboard: Streamlit")