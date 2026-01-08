import time

FEATURE_COUNT = 5

press_time = {}

last_press_time = None
last_release_time = None


def extract_realtime_features(key: str, event_type: str, timestamp: float):
    global last_press_time, last_release_time

    key_str = str(key)

    # key_code: stable float fingerprint
    key_code = float(abs(hash(key_str)) % 10000) / 10000.0

    # ----------------- KEY DOWN -----------------
    if event_type == "down":
        press_time[key_str] = timestamp

        # Down→Down latency
        dd = 0.0
        if last_press_time is not None:
            dd = timestamp - last_press_time

        last_press_time = timestamp

        # do NOT return features yet (only on key release)
        return None

    # ----------------- KEY UP -----------------
    if event_type == "up":
        # Hold time
        hold = 0.0
        if key_str in press_time:
            hold = timestamp - press_time[key_str]

        # Up→Up latency
        uu = 0.0
        if last_release_time is not None:
            uu = timestamp - last_release_time

        # UD latency approximation
        ud = uu

        # Down→Down latency recomputed after release
        dd = 0.0
        if last_press_time is not None:
            dd = timestamp - last_press_time

        # Clamp negatives and cap extreme values to improve robustness
        def clamp(v):
            if v < 0:
                v = 0.0
            if v > 2.0:
                v = 2.0
            return v

        hold = clamp(hold)
        uu = clamp(uu)
        ud = clamp(ud)
        dd = clamp(dd)

        last_release_time = timestamp

        # Return exactly 5 features in the same order every time
        return [
            key_code,      # 0
            hold,          # 1
            ud,            # 2
            dd,            # 3
            uu             # 4
        ]