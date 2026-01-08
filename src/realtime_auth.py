import numpy as np
import time
from collections import deque
from pynput import keyboard
from tensorflow.keras.models import load_model
from realtime_features import extract_realtime_features, FEATURE_COUNT
import argparse


# -------------------------------
# CONFIG
# -------------------------------
WINDOW = 20
window = []
pred_buf = deque(maxlen=5)

# -------------------------------
# Args
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--one-shot", action="store_true")
args = parser.parse_known_args()[0]

one_shot = args.one_shot
silent = False     # ALWAYS SHOW EVENTS FOR DEBUG


# -------------------------------
# Load model
# -------------------------------
print("📥 Loading model...")
model = load_model("models/lstm.keras")
print("✅ Model loaded")


# -------------------------------
# Load normalization stats
# -------------------------------
mean = np.zeros(FEATURE_COUNT)
std = np.ones(FEATURE_COUNT)

try:
    norm = np.load("models/feature_norm.npz")
    mean = norm["mean"]
    std = norm["std"]
    std[std == 0] = 1
    print("📊 Loaded normalization stats")
except:
    print("⚠️ No normalization found, using raw features")


# -------------------------------
# Load threshold
# -------------------------------
threshold = 0.80
try:
    with open("models/threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    print(f"🎯 Loaded threshold = {threshold:.2f}")
except:
    print("⚠️ No threshold found, default = 0.80")


print("\n🔐 Real-time authentication started")
if one_shot:
    print("🎯 ONE-SHOT MODE — will stop after 20 frames\n")
else:
    print("🎯 CONTINUOUS MODE — press ESC to exit\n")


# -------------------------------
# PREDICT
# -------------------------------
def make_prediction():
    X = np.array(window)[-WINDOW:]
    Xn = (X - mean) / std
    Xn = Xn.reshape(1, WINDOW, FEATURE_COUNT)

    p = float(model.predict(Xn, verbose=0)[0][0])

    # ONE-SHOT
    if one_shot:
        verdict = "🟢 USER" if p >= threshold else "❌ IMPOSTOR"
        print(f"\n=== FINAL DECISION ===\n{verdict}  (p={p:.3f}, thr={threshold:.2f})\n")
        return True

    # CONTINUOUS
    pred_buf.append(p)
    vote = float(np.median(pred_buf))
    verdict = "🟢 OK" if vote >= threshold else "❌ IMPOSTOR"
    print(f"{verdict} (p={p:.3f}, vote={vote:.3f}, thr={threshold:.2f})")
    return False


# -------------------------------
# KEY PRESS
# -------------------------------
def on_press(key):
    ts = time.time()

    # ESC stops always
    if key == keyboard.Key.esc:
        if one_shot and len(window) < WINDOW:
            print("❌ NOT ENOUGH DATA — need 20 release events")
        print("🛑 STOP")
        return False

    try:
        k = key.char
    except:
        k = str(key)

    extract_realtime_features(k, "down", ts)


# -------------------------------
# KEY RELEASE
# -------------------------------
def on_release(key):
    ts = time.time()

    try:
        k = key.char
    except:
        k = str(key)

    fv = extract_realtime_features(k, "up", ts)

    if fv is None:
        return

    window.append(fv)

    # sliding window
    if len(window) > WINDOW:
        window.pop(0)

    # DEBUG print
    print(f"[RELEASE] {k} | hold={fv[1]*1000:.1f} ms | window={len(window)}")

    # only predict when window is full
    if len(window) == WINDOW:
        should_stop = make_prediction()
        if should_stop:
            return False

    if key == keyboard.Key.esc:
        return False


# -------------------------------
# RUN LISTENER
# -------------------------------
with keyboard.Listener(on_press=on_press, on_release=on_release) as L:
    L.join()