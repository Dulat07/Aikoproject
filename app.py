import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import os

SEQ_LEN = 20
FEATURES = 5
lstm_buffer = []
pred_buf = deque(maxlen=5)

lstm_model = load_model("models/lstm.keras")

mean = np.zeros(FEATURES)
std = np.ones(FEATURES)
try:
    norm = np.load("models/feature_norm.npz")
    mean = norm["mean"]
    std = norm["std"]
    std[std == 0] = 1.0
except Exception:
    pass

threshold = 0.80
try:
    with open("models/threshold.txt", "r") as f:
        threshold = float(f.read().strip())
except Exception:
    pass

def authenticate_user(event_row):
    v = np.array(event_row, dtype=float)
    lstm_buffer.append(v)
    if len(lstm_buffer) > SEQ_LEN:
        lstm_buffer.pop(0)
    if len(lstm_buffer) < SEQ_LEN:
        return None
    window = np.array(lstm_buffer)
    Xn = (window - mean) / std
    Xn = Xn.reshape(1, SEQ_LEN, FEATURES)
    p = lstm_model.predict(Xn, verbose=0)[0][0]
    pred_buf.append(p)
    vote = float(np.median(np.array(pred_buf)))
    return vote >= threshold

if __name__ == "__main__":
    print("🚀 LSTM Authentication Demo")
    example_event = [0.5, 0.05, 0.05, 0.05, 0.05]
    result = authenticate_user(example_event)
    if result:
        print("✔ User verified")
    else:
        print("❌ Access denied (behavior mismatch)")