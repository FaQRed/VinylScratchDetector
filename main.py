import os
import io
import torch
import librosa
from librosa import feature
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from transformers import Wav2Vec2FeatureExtractor, AutoModel

app = FastAPI(title="Vinyl Scratch Detector API")


WINDOW_SEC = 3.0
HOP_SEC = 1.0
THRESHOLD = 0.7
FEATURE_NAMES = ['zcr_mean', 'zcr_std', 'zcr_max', 'rmse_mean', 'rmse_max',
                 'mfcc_mean', 'centroid_mean', 'centroid_max']

# mac device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# MERT
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
mert_encoder = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
mert_tf = tf.keras.models.load_model('models/with_mert/vinyl_mert_classifier.keras')

# classic
rf_model = joblib.load('models/random_forest_model/rf_model.pkl')
svm_model = joblib.load('models/svm_model/svm_model.pkl')
scaler = joblib.load('models/random_forest_model/scaler.pkl')

# CNN
cnn_model = tf.keras.models.load_model('models/cnn/vinyl_cnn_model.keras')


def extract_segment_features(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return [
        np.mean(zcr), np.std(zcr), np.max(zcr),
        np.mean(rmse), np.max(rmse),
        np.mean(mfccs),
        np.mean(centroid), np.max(centroid)
    ]


async def load_audio_from_upload(file, sr):
    data = await file.read()
    y, _ = librosa.load(io.BytesIO(data), sr=sr)
    return y


def clean_results(scratches):
    if not scratches:
        return {"status": "clean",
                "total": 0,
                "seconds": []}

# duplicates filter
    filtered = []
    last_s = -10
    for s in scratches:
        if s["second"] - last_s > 1.5:
            filtered.append(s)
            last_s = s["second"]

    return {"status": "scratch_detected",
            "total": len(filtered),
            "seconds": filtered}


@app.post("/analyze/mert")
async def analyze_mert(file: UploadFile = File(...)):
    sr = 24000 # Hgz
    y = await load_audio_from_upload(file, sr)
    found = []

    step, window = int(HOP_SEC * sr), int(WINDOW_SEC * sr)
    for i in range(0, len(y) - window, step):
        segment = y[i:i + window]
        inputs = processor(segment, sampling_rate=sr, return_tensors="pt").to(device)

        with torch.no_grad():
            out = mert_encoder(**inputs, output_hidden_states=True)

        # 1536 features
        max_pool = torch.max(out.last_hidden_state, dim=1).values
        mean_pool = torch.mean(out.hidden_states[1], dim=1)
        feat = torch.cat((max_pool, mean_pool), dim=1).cpu().numpy()


        prob = float(mert_tf.predict(feat, verbose=0)[0][0])
        if prob > THRESHOLD:
            found.append({"second": round(i / sr, 2),
                          "probability": round(prob, 3)})

    return clean_results(found)


@app.post("/analyze/rf")
async def analyze_rf(file: UploadFile = File(...)):
    sr = 16000
    y = await load_audio_from_upload(file, sr)
    found = []

    step, win = int(HOP_SEC * sr), int(WINDOW_SEC * sr)
    for i in range(0, len(y) - win, step):
        segment = y[i:i + win]
        raw_feat = extract_segment_features(segment, sr)


        df_feat = pd.DataFrame([raw_feat], columns=FEATURE_NAMES)
        prob = rf_model.predict_proba(scaler.transform(df_feat))[0][1]

        if prob > THRESHOLD:
            found.append({"second": round(i / sr, 2),
                          "probability": round(prob, 3)})

    return clean_results(found)


@app.post("/analyze/svm")
async def analyze_svm(file: UploadFile = File(...)):
    sr = 16000
    y = await load_audio_from_upload(file, sr)
    found = []

    step, win = int(HOP_SEC * sr), int(WINDOW_SEC * sr)
    for i in range(0, len(y) - win, step):
        segment = y[i:i + win]
        raw_feat = extract_segment_features(segment, sr)
        df_feat = pd.DataFrame([raw_feat], columns=FEATURE_NAMES)
        prob = svm_model.predict_proba(scaler.transform(df_feat))[0][1]

        if prob > THRESHOLD:
            found.append({"second": round(i / sr, 2),
                          "probability": round(prob, 3)})

    return clean_results(found)


@app.post("/analyze/cnn")
async def analyze_cnn(file: UploadFile = File(...)):
    sr = 16000 # was trained on 16 KHz
    y = await load_audio_from_upload(file, sr)
    found = []

    step, win = int(HOP_SEC * sr), int(WINDOW_SEC * sr)

    for i in range(0, len(y) - win, step):
        segment = y[i:i + win]

        #MelSpec generation
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=512, hop_length=256, n_mels=96)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalization (finding the quitest and the loudest sound)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)


        # (1, 96, 188, 1)
        input_data = np.expand_dims(mel_norm, axis=(0, -1))


        prob = float(cnn_model.predict(input_data, verbose=0)[0][0])

        if prob > THRESHOLD:
            found.append({"second": round(i / sr, 2),
                          "probability": round(prob, 3)})

    return clean_results(found)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)