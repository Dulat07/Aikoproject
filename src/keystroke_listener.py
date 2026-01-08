from pynput import keyboard
import time

last_press_time = None

def on_press(key):
    global last_press_time

    t = time.time()

    try:
        k = key.char
    except:
        k = str(key)

    print(f"[DOWN] {k} at {t}")

    last_press_time = t


def on_release(key):
    global last_press_time

    t = time.time()

    try:
        k = key.char
    except:
        k = str(key)

    if last_press_time:
        hold = t - last_press_time
        print(f"[UP]   {k} at {t} | hold={hold:.4f}s")

    # ESC to stop
    if key == keyboard.Key.esc:
        print("Stopping listener...")
        return False


if __name__ == "__main__":
    print("Listening for keystrokes... (press ESC to stop)")
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as listener:
        listener.join()