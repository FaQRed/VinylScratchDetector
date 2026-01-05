import os
import pandas as pd
import glob


def build_dataset_df(data_path):
    file_paths = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)

    data_list = []

    for path in file_paths:
        filename = os.path.basename(path)

        # sect1 -> 1 (scratch), sect0 -> 0 (clean)
        if 'sect1' in filename:
            label = 1
        elif 'sect0' in filename:
            label = 0
        else:
            continue

        data_list.append({
            'file_path': path,
            'label': label,
        })

    return pd.DataFrame(data_list)


if __name__ == "__main__":

    df = build_dataset_df('../data')


    print(df['label'].value_counts())
    print(df.head())