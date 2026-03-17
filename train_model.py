import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import pickle

# ------------------------------
# DATASET PATH
# ------------------------------
dataset_path = "dataset"
classes = ["Microplastic", "Non-Microplastic"]

data = []
labels = []

# ------------------------------
# LOAD IMAGES
# ------------------------------
for label, category in enumerate(classes):
    folder = os.path.join(dataset_path, category)

    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((64, 64))
            img = np.array(image) / 255.0
            img = img.flatten()

            data.append(img)
            labels.append(label)
        except:
            continue

data = np.array(data)
labels = np.array(labels)

print(f"✅ Total images loaded: {len(data)}")

# ------------------------------
# TRAIN MODEL
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data, labels)

# ------------------------------
# SAVE MODEL
# ------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 model.pkl created successfully!")