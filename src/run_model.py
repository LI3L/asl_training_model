"""
run_model.py — Run the model on the laptop webcam in place of the ESP32-S3
Sense (deploy/esp32/ASL_Detector/ASL_Detector.ino): prints predictions to
stdout in the exact same log format the device does, AND, if --car is given,
drives the Arduino Mega car controller directly over serial -- doing the job
of serial_bridge.py's USB bridge (Option 2 in ASL_Car_Controller.ino) itself,
so you don't need to run that script separately.

Reuses model loading/preprocessing from test_model.py and shows the same
"ASL Live Inference" + "Model Input (28x28 preprocessed)" windows.

Usage:
    python run_model.py                       # tflite model, logs to stdout only
    python run_model.py --keras               # use the .keras model instead
    python run_model.py --car                  # also drive the Mega: auto-picks
                                               # a wired port (/dev/ttyACM*,
                                               # /dev/ttyUSB*) if one is plugged
                                               # in, otherwise falls back to a
                                               # paired Bluetooth SPP port
                                               # (/dev/rfcomm*) -- see the main
                                               # README's "ASL Car Control"
                                               # section for pairing/binding
                                               # the Bluetooth module first.
    python run_model.py --car /dev/ttyACM0     # skip auto-detection, use this
                                               # exact port (wired or BT alike)
    python run_model.py --car --threshold 80

Sends a single letter (W/Y/C/A/O) at 9600 baud whenever the stable prediction
is a car command above --threshold confidence.

Press 'q' to quit, 's' to save the current frame for debugging.
"""

import glob
import os
import sys
import time

import cv2

from test_model import (
    ALPHABET,
    load_keras_model,
    load_tflite_model,
    predict,
    preprocess_image,
)

# Mirrors the constants in deploy/esp32/ASL_Detector/ASL_Detector.ino so the
# logged output matches the device exactly.
MIN_CONFIDENCE = 0.40
SMOOTH_WINDOW = 5
SMOOTH_MAJORITY = 3

# Matches CAR_CONFIDENCE_THRESHOLD and isCarCommand in ASL_Detector.ino, and
# the FORWARD/BACKWARD/LEFT/RIGHT/STOP letters ASL_Car_Controller.ino acts on.
CAR_CONFIDENCE_THRESHOLD = 80.0
CAR_COMMAND_LETTERS = {"W", "Y", "C", "A", "O"}
CAR_BAUD = 9600

# Auto-detection order for --car with no explicit port: a wired connection is
# preferred (lower latency, no pairing needed), a paired Bluetooth SPP module
# (see main README's "ASL Car Control" section for pairing/binding it to one
# of these paths) is the fallback for when no cable is plugged in.
CAR_WIRED_PORT_GLOBS = ["/dev/ttyACM*", "/dev/ttyUSB*"]
CAR_BT_PORT_GLOBS = ["/dev/rfcomm*"]


def find_car_port(explicit_port):
    if explicit_port is not None:
        return explicit_port, "explicit"
    for pattern in CAR_WIRED_PORT_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0], "wired"
    for pattern in CAR_BT_PORT_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0], "bluetooth"
    return None, None


def make_log_line(elapsed_ms, letter, confidence_pct):
    if letter is None:
        return f"[{elapsed_ms} ms] ...\n"
    return f"[{elapsed_ms} ms] letter={letter}  confidence={confidence_pct:.1f}%\n"


def run(model, is_tflite, car_port):
    print("Starting webcam... Press 'q' to quit, 's' to save the current frame for debugging.")
    save_dir = "/app/output/debug_frames"
    os.makedirs(save_dir, exist_ok=True)
    save_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open /dev/video0. Check permissions.")
        return

    # Ring buffer of recent above-threshold predictions, used for the same
    # majority-vote smoothing the ESP32 firmware does before it logs a line.
    history = [-1] * SMOOTH_WINDOW
    history_conf = [0.0] * SMOOTH_WINDOW
    idx = 0

    start = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            cv2.rectangle(
                frame,
                (width // 2 - 150, height // 2 - 150),
                (width // 2 + 150, height // 2 + 150),
                (0, 255, 0),
                2,
            )
            roi = frame[
                height // 2 - 150 : height // 2 + 150,
                width // 2 - 150 : width // 2 + 150,
            ]

            input_data = preprocess_image(roi)
            letter, confidence = predict(model, input_data, is_tflite)

            if confidence >= MIN_CONFIDENCE:
                history[idx] = ALPHABET.index(letter)
                history_conf[idx] = confidence
                idx = (idx + 1) % SMOOTH_WINDOW

            counts = [0] * len(ALPHABET)
            conf_sum = [0.0] * len(ALPHABET)
            for h, c in zip(history, history_conf):
                if h >= 0:
                    counts[h] += 1
                    conf_sum[h] += c

            stable_letter_idx = max(range(len(ALPHABET)), key=lambda i: counts[i])
            stable_count = counts[stable_letter_idx]

            elapsed_ms = int((time.monotonic() - start) * 1000)
            stable_letter = None
            stable_conf_pct = 0.0
            if stable_count >= SMOOTH_MAJORITY:
                stable_letter = ALPHABET[stable_letter_idx]
                stable_conf_pct = (conf_sum[stable_letter_idx] / stable_count) * 100.0
            log_line = make_log_line(elapsed_ms, stable_letter, stable_conf_pct)

            sys.stdout.write(log_line)
            sys.stdout.flush()

            # Same gating as ASL_Detector.ino's isCarCommand check before it
            # forwards to the Mega: only the 5 drive letters, only once past
            # the (stricter, separate) car confidence threshold.
            if (
                car_port is not None
                and stable_letter in CAR_COMMAND_LETTERS
                and stable_conf_pct >= CAR_CONFIDENCE_THRESHOLD
            ):
                car_port.write(f"{stable_letter}\n".encode())

            # Raw per-frame prediction (matches test_model.py's overlay).
            cv2.putText(
                frame,
                f"Pred: {letter} ({confidence * 100:.1f}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
            # Smoothed value that was actually logged this frame, i.e. what
            # the ESP32 would have sent to serial_bridge.py.
            cv2.putText(
                frame,
                f"Log: {log_line.strip()}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )

            cv2.imshow("ASL Live Inference", frame)

            debug_view = (input_data[0, :, :, 0] * 255).astype("uint8")
            debug_view = cv2.resize(debug_view, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Model Input (28x28 preprocessed)", debug_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f"{save_dir}/roi_{save_count:03d}_pred_{letter}.png", roi)
                cv2.imwrite(f"{save_dir}/model_input_{save_count:03d}_pred_{letter}.png", debug_view)
                print(f"Saved frame {save_count} (predicted {letter})")
                save_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    use_keras = "--keras" in sys.argv

    car_requested = "--car" in sys.argv
    car_arg = None
    if car_requested:
        idx = sys.argv.index("--car") + 1
        if idx < len(sys.argv) and not sys.argv[idx].startswith("--"):
            car_arg = sys.argv[idx]

    if "--threshold" in sys.argv:
        CAR_CONFIDENCE_THRESHOLD = float(sys.argv[sys.argv.index("--threshold") + 1])

    if use_keras:
        model_path = "/app/output/asl_model.keras"
        if not os.path.exists(model_path):
            print(f"Error: Keras model not found at {model_path}. Run the training script first.")
            sys.exit(1)
        model = load_keras_model(model_path)
        is_tflite = False
    else:
        model_path = "/app/output/asl_model.tflite"
        if not os.path.exists(model_path):
            print(f"Error: TFLite model not found at {model_path}. Run the conversion script first.")
            sys.exit(1)
        model = load_tflite_model(model_path)
        is_tflite = True

    car_port = None
    if car_requested:
        try:
            import serial
        except ImportError:
            print("ERROR: pyserial is not installed. Run:")
            print("  pip install pyserial")
            sys.exit(1)

        port_path, kind = find_car_port(car_arg)
        if port_path is None:
            print("ERROR: No car controller found.")
            print(f"  Looked for a wired port matching {CAR_WIRED_PORT_GLOBS},")
            print(f"  then a paired Bluetooth port matching {CAR_BT_PORT_GLOBS}.")
            print("  Plug in the Mega over USB, or pair + bind the Bluetooth")
            print("  module first (see the main README's 'ASL Car Control' section):")
            print("    sudo rfcomm bind rfcomm0 <MAC_ADDRESS> 1")
            sys.exit(1)

        print(f"Opening Arduino Mega on {port_path} ({kind}) at {CAR_BAUD} baud...")
        car_port = serial.Serial(port_path, CAR_BAUD, timeout=1)
        # Wait for the Mega to reset after the serial connection opens,
        # same as serial_bridge.py does.
        time.sleep(2)
        print(f"  Connected. Forwarding W/Y/C/A/O commands >= {CAR_CONFIDENCE_THRESHOLD}% confidence.")

    try:
        run(model, is_tflite, car_port)
    finally:
        if car_port is not None:
            car_port.close()
