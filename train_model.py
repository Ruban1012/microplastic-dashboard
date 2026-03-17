import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

data = []
labels = []

dataset_path = "dataset"
classes = ["Microplastic", "Non-Microplastic"]

for label, category in enumerate(classes):
    folder = os.path.join(dataset_path, category)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = img.flatten()

        data.append(img)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(data, labels)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")