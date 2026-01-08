import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import glob
import os

WINDOW = 20
FEATURES = 5

def load_all_csv():
    X, y = [], []
    for path in glob.glob("collector/saved/user_*.csv"):
        df = pd.read_csv(path, header=None)
        arr = df.values
        for i in range(len(arr)-WINDOW+1):
            X.append(arr[i:i+WINDOW])
            y.append(1)
    for path in glob.glob("collector/saved/impostor_*.csv"):
        df = pd.read_csv(path, header=None)
        arr = df.values
        for i in range(len(arr)-WINDOW+1):
            X.append(arr[i:i+WINDOW])
            y.append(0)
    return np.array(X), np.array(y)

def compute_norm_stats(X):
    # X shape: (N, WINDOW, FEATURES)
    flat = X.reshape(-1, FEATURES)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def apply_norm(X, mean, std):
    return (X - mean) / std

def calibrate_threshold(model, X_val, y_val):
    # Choose threshold that maximizes Youden's J (TPR-FPR)
    preds = model.predict(X_val, verbose=0).reshape(-1)
    thresholds = np.linspace(0.3, 0.95, 66)
    best_t = 0.8
    best_j = -1
    for t in thresholds:
        y_hat = (preds >= t).astype(int)
        tp = ((y_hat == 1) & (y_val == 1)).sum()
        fp = ((y_hat == 1) & (y_val == 0)).sum()
        fn = ((y_hat == 0) & (y_val == 1)).sum()
        tn = ((y_hat == 0) & (y_val == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)

def train():
    os.makedirs("models", exist_ok=True)

    X, y = load_all_csv()
    print("X:", X.shape, "y:", y.shape)

    mean, std = compute_norm_stats(X)
    np.savez("models/feature_norm.npz", mean=mean, std=std)

    Xn = apply_norm(X, mean, std)

    X_train, X_val, y_train, y_val = train_test_split(
        Xn, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Input(shape=(WINDOW, FEATURES)),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    cp = ModelCheckpoint("models/lstm.keras", save_best_only=True, monitor="val_loss")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[cp, es])

    thr = calibrate_threshold(model, X_val, y_val)
    with open("models/threshold.txt", "w") as f:
        f.write(str(thr))

    print("Model saved! Normalization and threshold stored.")

if __name__ == "__main__":
    train()