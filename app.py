import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
import joblib
from PIL import Image

# Page settings
st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🌊 Microplastic Detection Dashboard")
st.write("Upload a microscopic image to detect microplastics")

# Load models
@st.cache_resource
def load_models():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    cnn_model = Model(inputs=base_model.input, outputs=base_model.output)

    rf_model = joblib.load("rf_model.pkl")  # Load your trained model
    return cnn_model, rf_model

cnn_model, rf_model = load_models()

# Preprocess image
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)
    processed = preprocess(img_np)

    # Feature extraction
    features = cnn_model.predict(processed)

    # Prediction
    prediction = rf_model.predict(features)[0]
    probabilities = rf_model.predict_proba(features)

    confidence = np.max(probabilities) * 100

    classes = ["Microplastic", "Non-Microplastic"]

    # Display result
    st.subheader("🔍 Prediction Result")
    st.success(f"Class: {classes[int(prediction)]}")
    st.metric("Confidence", f"{confidence:.2f}%")

st.write("---")
st.write("Model: ResNet + Random Forest | Dashboard: Streamlit")