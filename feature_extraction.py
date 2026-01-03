import pandas as pd
import librosa
import librosa.feature as ft
import numpy as np
from tqdm import tqdm

from preprocess import build_dataset_df


def extract_all_features(file_path):
    try:

        y, sr = librosa.load(file_path, duration=3.0)


        zcr = ft.zero_crossing_rate(y)
        rmse = ft.rms(y=y)
        mfccs = ft.mfcc(y=y, sr=sr, n_mfcc=13)
        centroid = ft.spectral_centroid(y=y, sr=sr)


        features = {
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'zcr_max': np.max(zcr),
            'rmse_mean': np.mean(rmse),
            'rmse_max': np.max(rmse),
            'mfcc_mean': np.mean(mfccs),
            'centroid_mean': np.mean(centroid),
            'centroid_max': np.max(centroid)
        }
        return features
    except Exception as e:
        print(f"Error while reading {file_path}: {e}")
        return None


if __name__ == "__main__":
    print("Scanning the folder with data...")
    df = build_dataset_df('./data')

    if df.empty:
        print("Error: DataFrame is empty. Check the path to './data'")
    else:

        data_list = []
        print(f"Extracting features from {len(df)} files...")

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            feat = extract_all_features(row['file_path'])
            if feat:
                feat['label'] = row['label']
                data_list.append(feat)


        df_features = pd.DataFrame(data_list)


        df_features['file_path'] = df['file_path']

        df_features.to_csv('extracted_features.csv', index=False)
        print("\nDone! Features with metadata are saved to 'extracted_features.csv'")