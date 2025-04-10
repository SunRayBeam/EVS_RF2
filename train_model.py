import os
import joblib
import librosa
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
CSV_PATH = "PATH_TO_ESC50/meta/esc50.csv"
AUDIO_DIR = "PATH_TO_ESC50/audio"
MODULE_DIR = "evs_rf2"
MODEL_PATH = os.path.join(MODULE_DIR, "model.pkl")

# Create module directory if not exists
os.makedirs(MODULE_DIR, exist_ok=True)

# Load ESC-50 metadata
df = pd.read_csv(CSV_PATH)

# Feature extractor
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=48000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    features = np.concatenate([
        mfcc.mean(axis=1),
        [spectral_centroid.mean()],
        [spectral_bandwidth.mean()],
        [zero_crossing.mean()]
    ])
    return features

# Extract features
X, y = [], []
for _, row in df.iterrows():
    features = extract_features(os.path.join(AUDIO_DIR, row["filename"]))
    X.append(features)
    y.append(row["category"])

# Train model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(np.array(X), np.array(y))
joblib.dump(pipeline, MODEL_PATH)

# Write utils
with open(os.path.join(MODULE_DIR, "model_utils.py"), "w") as f:
    f.write(f'''import os
import librosa
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=48000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    features = np.concatenate([
        mfcc.mean(axis=1),
        [spectral_centroid.mean()],
        [spectral_bandwidth.mean()],
        [zero_crossing.mean()]
    ])
    return features

def predict_category(file_path):
    features = extract_features(file_path).reshape(1, -1)
    return model.predict(features)[0]
''')

print("✅ Model trained and saved to evs_rf2/model.pkl")
