import streamlit as st
import numpy as np
from PIL import Image
import pickle

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🌊 Microplastic Detection Dashboard")
st.write("Upload a microscopic image to detect microplastics")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------------------
# CLASS LABELS
# ------------------------------
classes = ["Microplastic", "Non-Microplastic"]

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ------------------------------
    # IMAGE PROCESSING (NO CV2)
    # ------------------------------
    image = image.resize((64, 64))   # Resize using PIL
    img = np.array(image) / 255.0    # Normalize

    # Flatten
    img_flat = img.flatten().reshape(1, -1)

    # ------------------------------
    # PREDICTION
    # ------------------------------
    prediction = model.predict(img_flat)[0]
    confidence = np.max(model.predict_proba(img_flat)) * 100

    # ------------------------------
    # OUTPUT
    # ------------------------------
    st.subheader("🔍 Prediction Result")

    if prediction == 0:
        st.success(f"Class: {classes[0]}")
    else:
        st.error(f"Class: {classes[1]}")

    st.metric("Confidence", f"{confidence:.2f}%")

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.write("Model: Random Forest | Dashboard: Streamlit")