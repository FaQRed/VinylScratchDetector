import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def get_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    X = []
    for path in df['npy_path']:
        X.append(np.load(path))
    X = np.array(X)[..., np.newaxis]
    y = df['label'].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Data ready: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_test, X_val, y_train, y_test, y_val