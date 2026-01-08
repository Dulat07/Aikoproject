import pandas as pd
import numpy as np
from feature_extractor import extract_features_from_raw


def preprocess_lstm(csv_path, label_value=1, seq_len=20):
    raw = pd.read_csv(csv_path)
    df_feat = extract_features_from_raw(raw)

    X = create_sequences(df_feat, seq_len)
    y = np.full(len(X), label_value)

    return X, y


def create_sequences(df, seq_len):
    X = []

    for i in range(len(df) - seq_len + 1):
        window = df.iloc[i:i+seq_len].values
        window_norm = normalize_window(window)
        X.append(window_norm)

    return np.array(X)


def normalize_window(window):
    min_vals = window.min(axis=0)
    max_vals = window.max(axis=0)
    dif = max_vals - min_vals
    dif[dif == 0] = 1
    return (window - min_vals) / dif