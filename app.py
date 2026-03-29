import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
import pandas as pd

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Microplastic AI Dashboard", layout="wide")

# ------------------------------
# TITLE
# ------------------------------
st.title("🌊 Microplastic Detection System")
st.markdown("### AI-based Microplastic Classification Dashboard")

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
# SIDEBAR (DATASET CONTROL)
# ------------------------------
st.sidebar.header("⚙️ Controls")

dataset_path = "dataset"
selected_class = st.sidebar.selectbox("Select Category", classes)

folder_path = os.path.join(dataset_path, selected_class)
images = os.listdir(folder_path)

selected_image = st.sidebar.selectbox("Select Image", images)

# ------------------------------
# MAIN LAYOUT (2 COLUMNS)
# ------------------------------
col1, col2 = st.columns(2)

# ------------------------------
# LEFT SIDE → IMAGE
# ------------------------------
with col1:
    st.subheader("📸 Selected Image")

    img_path = os.path.join(folder_path, selected_image)
    image = Image.open(img_path).convert("RGB")
    st.image(image, use_column_width=True)

# ------------------------------
# RIGHT SIDE → RESULT
# ------------------------------
with col2:
    st.subheader("🔍 Prediction Result")

    # Process image
    image_resized = image.resize((64, 64))
    img = np.array(image_resized) / 255.0
    img_flat = img.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(img_flat)[0]
    probs = model.predict_proba(img_flat)[0]
    confidence = np.max(probs) * 100

    # Display result
    if prediction == 0:
        st.success(f"✅ {classes[0]}")
    else:
        st.error(f"❌ {classes[1]}")

    st.metric("Confidence", f"{confidence:.2f}%")

    # ------------------------------
    # PROBABILITY CHART
    # ------------------------------
    st.subheader("📊 Prediction Analysis")

    chart_data = pd.DataFrame({
        "Class": classes,
        "Probability": probs
    })

    st.bar_chart(chart_data.set_index("Class"))

    # ------------------------------
    # EXTRA DETAILS
    # ------------------------------
    st.subheader("📈 Detailed Values")

    st.write(f"Microplastic: {probs[0]*100:.2f}%")
    st.write(f"Non-Microplastic: {probs[1]*100:.2f}%")

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.markdown("💡 **Model:** Random Forest  |  **UI:** Streamlit Dashboard")