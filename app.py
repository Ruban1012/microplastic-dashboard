import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Microplastic Detection", layout="centered")

st.title("🌊 Microplastic Detection Dashboard")

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
# DATASET MODE
# ------------------------------
dataset_path = "dataset"

selected_class = st.selectbox("Select Category", classes)
folder_path = os.path.join(dataset_path, selected_class)

images = os.listdir(folder_path)
selected_image = st.selectbox("Select Image", images)

img_path = os.path.join(folder_path, selected_image)

image = Image.open(img_path).convert("RGB")
st.image(image, caption="Selected Image", use_column_width=True)

# ------------------------------
# PROCESS IMAGE
# ------------------------------
image = image.resize((64, 64))
img = np.array(image) / 255.0
img_flat = img.flatten().reshape(1, -1)

# ------------------------------
# PREDICTION
# ------------------------------
prediction = model.predict(img_flat)[0]
probs = model.predict_proba(img_flat)[0]
confidence = np.max(probs) * 100

# ------------------------------
# RESULT
# ------------------------------
st.subheader("🔍 Prediction Result")

if prediction == 0:
    st.success(f"Class: {classes[0]}")
else:
    st.error(f"Class: {classes[1]}")

st.metric("Confidence", f"{confidence:.2f}%")

# ------------------------------
# 📊 BAR CHART (IMPORTANT)
# ------------------------------
st.subheader("📊 Prediction Probability")

fig, ax = plt.subplots()
ax.bar(classes, probs)
ax.set_ylabel("Probability")
ax.set_title("Class Prediction Probability")

st.pyplot(fig)

# ------------------------------
# 📈 EXTRA INFO
# ------------------------------
st.subheader("📈 Analysis")

st.write(f"Microplastic Probability: {probs[0]*100:.2f}%")
st.write(f"Non-Microplastic Probability: {probs[1]*100:.2f}%")

st.write("---")
st.write("Model: Random Forest | Visualization Enabled")