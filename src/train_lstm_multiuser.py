import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from lstm_preprocess import preprocess_lstm
from model import build_lstm

DATA_DIR = "data/raw"
TARGET_USER = "participant1"  # whoever is the positive class


def load_all_users(data_dir):
    """
    Load every CSV in data/raw and assign labels:
    - participant1.csv -> label = 1
    - all others       -> label = 0
    """
    sequences_total = []
    labels_total = []

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    print("📂 Found CSV files:", files)

    for fname in files:
        user_path = os.path.join(data_dir, fname)
        username = fname.split(".")[0]

        label = 1 if TARGET_USER in username else 0

        print(f"➡ Processing {fname}, label={label}")

        try:
            X, y = preprocess_lstm(user_path, label_value=label, seq_len=20)
            sequences_total.append(X)
            labels_total.append(y)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

    if not sequences_total:
        raise RuntimeError("No valid CSV files found to train the model!")

    X = np.concatenate(sequences_total, axis=0)
    y = np.concatenate(labels_total, axis=0)

    print(f"\n🔢 FINAL SHAPES:")
    print("X =", X.shape)
    print("y =", y.shape)

    return X, y


def train_lstm_multiuser(epochs=25, save_path="models/lstm_multi.keras"):
    print("\n📥 Loading all datasets...")
    X, y = load_all_users(DATA_DIR)

    print("\n🧠 Building model...")
    model = build_lstm(input_shape=(X.shape[1], X.shape[2]))

    os.makedirs("models", exist_ok=True)

    checkpoint = ModelCheckpoint(
        save_path,
        save_best_only=True,
        monitor="loss",
        mode="min"
    )

    print("\n🔥 Training multi-user LSTM...")
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=8,
        callbacks=[checkpoint],
        verbose=1
    )

    print("\n💾 Model saved to:", save_path)
    return model


if __name__ == "__main__":
    train_lstm_multiuser()