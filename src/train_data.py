import pandas as pd


def load():
    train_path = 'data/train_data.csv'
    labels_path = 'data/train_labels.csv'

    train_df = pd.read_csv(train_path)
    labels_df = pd.read_csv(labels_path)

    merged = train_df.combine_first(labels_df)

    return merged
