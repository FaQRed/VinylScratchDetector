import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import pandas as pd
import glob
from tqdm import tqdm

DATA_PATH = '../../data'
SAVE_DIR = '../../mert_features/'
SR_MERT = 24000 # Hgz
WINDOW_SEC = 3.0
HOP_SEC = 1.5

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading MERT model...")
model_name = "m-a-p/MERT-v1-95M"
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
mert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mert_model.to(device)
mert_model.eval()


def extract_windows_features(file_path):
    audio, _ = librosa.load(file_path, sr=SR_MERT)
    samples_per_window = int(WINDOW_SEC * SR_MERT)
    samples_per_hop = int(HOP_SEC * SR_MERT)
    embeddings = []

    for start in range(0, len(audio) - samples_per_window, samples_per_hop):
        window = audio[start: start + samples_per_window]
        inputs = processor(window, sampling_rate=SR_MERT, return_tensors="pt").to(device) # will return tensors instead
        # of list of python integers. tf - Tensorflow tensors

        with torch.no_grad():
            outputs = mert_model(**inputs, output_hidden_states=True)

        feat_max = torch.max(outputs.last_hidden_state, dim=1).values
        feat_mean_early = torch.mean(outputs.hidden_states[1], dim=1)
        combined = torch.cat((feat_max, feat_mean_early), dim=1)
        embeddings.append(combined.squeeze().cpu().numpy())

    return embeddings


def main():
    file_paths = glob.glob(os.path.join(DATA_PATH, "**/*.wav"), recursive=True)
    data_log = []

    for path in tqdm(file_paths, desc="Progress"):
        filename = os.path.basename(path)

        if 'sect1' in filename:
            label = 1
        elif 'sect0' in filename:
            label = 0
        else:
            continue

        try:
            window_embs = extract_windows_features(path)
            for i, emb in enumerate(window_embs):
                feat_filename = f"{filename}_win_{i}.npy"
                save_path = os.path.join(SAVE_DIR, feat_filename)
                np.save(save_path, emb)
                data_log.append({'mert_path': save_path, 'label': label})
        except Exception as e:
            print(f"Error: {e}")

    if data_log:
        result_df = pd.DataFrame(data_log)
        result_df.to_csv('../../mert_dataset_map.csv', index=False)
        print("Saved!")
    else:
        print("No files found")


if __name__ == "__main__":
    main()