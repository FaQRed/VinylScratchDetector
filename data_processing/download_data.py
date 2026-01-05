import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_vinyl_dataset():
    api = KaggleApi()
    api.authenticate()

    dataset = "seandaly/detecting-scratch-noise-in-vinyl-playback"
    target_dir = "../data"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Downloading dataset {dataset}...")

    api.dataset_download_files(dataset, path=target_dir, unzip=True)
    print(f"Data saved to {target_dir}")

if __name__ == "__main__":
    download_vinyl_dataset()