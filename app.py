import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os

st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🌊 Microplastic Dataset Prediction")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

classes = ["Microplastic", "Non-Microplastic"]

# ------------------------------
# LOAD DATASET
# ------------------------------
dataset_path = "dataset"

selected_class = st.selectbox("Select Category", classes)

folder_path = os.path.join(dataset_path, selected_class)

images = os.listdir(folder_path)

# ------------------------------
# SELECT IMAGE FROM DATASET
# ------------------------------
selected_image = st.selectbox("Select Image", images)

img_path = os.path.join(folder_path, selected_image)

image = Image.open(img_path).convert("RGB")

st.image(image, caption="Dataset Image", use_column_width=True)

# ------------------------------
# PROCESS & PREDICT
# ------------------------------
image = image.resize((64, 64))
img = np.array(image) / 255.0
img_flat = img.flatten().reshape(1, -1)

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