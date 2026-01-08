import os
import joblib
from detector import extract_features
from sklearn.ensemble import RandomForestClassifier

X, y = [], []

for label, folder in [(0, "dataset/clean"), (1, "dataset/stego")]:
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        X.append(extract_features(path))
        y.append(label)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "stego_model.pkl")
print("✅ Model trained and saved.")
