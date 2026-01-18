import numpy as np
import librosa
from librosa import feature
import os
import pandas as pd
from tqdm import tqdm
import sys
from data_processing.preprocess import build_dataset_df

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)

def process_and_save_mel(file_path, save_path, sr=16000, duration=3.0):
    try:
        audio, _ = librosa.load(file_path, sr=sr, res_type='kaiser_fast') # resample type

        target_samples = int(duration * sr)
        if len(audio) >= target_samples:
            start = (len(audio) - target_samples) // 2
            audio = audio[start: start + target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

        mel_spec = feature.melspectrogram(
            y=audio, sr=sr, n_fft=512, hop_length=256, n_mels=96
        )

        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        np.save(save_path, mel_norm.astype(np.float32))
    except Exception as e:
        print(f"Error with file {file_path}: {e}")

if __name__ == "__main__":
    DATA_PATH = os.path.join(root_dir, 'data')
    OUTPUT_PATH = os.path.join(root_dir, 'processed_npy')

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    df = build_dataset_df(DATA_PATH)

    if df.empty:
        print("Error: DataFrame is empty. Check data folder")
        sys.exit(1)

    print(f"Started specs generation")

    dataset_map = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_name = f"label_{row['label']}_idx_{idx}.npy"
        save_path = os.path.join(OUTPUT_PATH, file_name)

        if process_and_save_mel(row['file_path'], save_path):
            dataset_map.append({
                'npy_path': save_path,
                'label': row['label']
            })

    map_df = pd.DataFrame(dataset_map)
    map_csv_path = os.path.join(root_dir, 'cnn_dataset_map.csv')
    map_df.to_csv(map_csv_path, index=False)

    print(f"Spectrograms (.npy) are saved to: {OUTPUT_PATH}")
    print(f"Datamap is saved to: {map_csv_path}")