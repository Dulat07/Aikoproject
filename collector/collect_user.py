import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from realtime_features import extract_realtime_features
import time
from pynput import keyboard
import csv

filename = f"collector/saved/user_{int(time.time())}.csv"
print("🟢 USER MODE — Type normally. ESC to stop.")
print(f"Saving to {filename}")

rows = []

def on_press(key):
    timestamp = time.time()
    try: k = key.char
    except: k = str(key)
    extract_realtime_features(k, "down", timestamp)

def on_release(key):
    timestamp = time.time()
    try: k = key.char
    except: k = str(key)

    fv = extract_realtime_features(k, "up", timestamp)
    if fv is not None:
        rows.append(fv)

    if key == keyboard.Key.esc:
        print("🛑 Stop")
        return False

with keyboard.Listener(on_press=on_press, on_release=on_release) as L:
    L.join()

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"💾 Saved user dataset: {filename}")
