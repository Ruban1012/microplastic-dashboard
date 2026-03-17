import os
import cv2
import numpy as np
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
# LOAD DATASET
# ------------------------------
for label, category in enumerate(classes):
    folder = os.path.join(dataset_path, category)

    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (64, 64))

        # Normalize
        img = img / 255.0

        # Flatten
        img = img.flatten()

        data.append(img)
        labels.append(label)

# Convert to numpy
data = np.array(data)
labels = np.array(labels)

print(f"✅ Total images loaded: {len(data)}")

# ------------------------------
# TRAIN MODEL
# ------------------------------
model = RandomForestClassifier(
    n_estimators=100,      # number of trees
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

model.fit(data, labels)

# ------------------------------
# SAVE MODEL
# ------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully!")