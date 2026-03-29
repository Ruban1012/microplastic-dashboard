import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🔬 Microplastic Detection Dashboard")
st.write("Upload a microscopic image to detect microplastics")

# ------------------------------
# LOAD MODEL (SAFE)
# ------------------------------
@st.cache_resource
def load_model():
    model_path = "rf_model.pkl"

    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# ------------------------------
# IMAGE PROCESSING FUNCTION
# ------------------------------
def preprocess_image(image):
    image = image.resize((64, 64))  # resize
    image = np.array(image)

    if len(image.shape) == 3:
        image = image.mean(axis=2)  # convert to grayscale

    image = image.flatten()  # flatten
    return image.reshape(1, -1)

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.error("❌ Model file not found! Upload 'rf_model.pkl' to GitHub.")
        st.stop()

    # preprocess
    features = preprocess_image(image)

    # prediction
    prediction = model.predict(features)[0]

    # result
    if prediction == 1:
        st.success("✅ Microplastic Detected")
    else:
        st.info("❌ No Microplastic Detected")

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.write("Model: Random Forest | Dashboard: Streamlit")