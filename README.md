# ESP32-S3 Sense: ASL TinyML Pipeline

This project is an automated Docker pipeline to train, fine-tune, and convert a custom American Sign Language (ASL) recognition model for the **Seeed Studio XIAO ESP32S3 Sense**.

It runs natively on Ubuntu Linux, leveraging your built-in webcam and X11 display server to test the model exactly as the microcontroller will see it, before compiling it into a C++ header file.

## 📁 Project Structure

```text
asl_training_model/
├── docker-compose.yml        # Docker environment configuration
├── Dockerfile                # OS and dependency build instructions
├── src/
│   ├── train_model.py        # Trains/fine-tunes the .keras model
│   ├── convert_model.py      # Converts .keras -> full int8 .tflite -> model.h
│   ├── test_model.py         # Live webcam / static-image OpenCV testing
│   ├── run_model.py          # Live webcam + ESP32-format logs; can drive the Mega directly (no ESP32 needed)
│   ├── capture_data.py       # Collect labeled real-hand samples from your webcam
│   ├── capture_backgrounds.py# Collect background-only photos (no hand) for augmentation
│   ├── augment_backgrounds.py# Composite real-hand samples onto varied backgrounds
│   └── serial_bridge.py      # USB bridge: forwards ESP32 predictions to the car
├── data/
│   ├── sign_mnist_train.csv  # Base Sign Language MNIST training set
│   ├── sign_mnist_valid.csv  # Base validation set
│   ├── custom_train.csv      # (generated) your real-hand samples from capture_data.py
│   ├── backgrounds/          # (generated) background-only photos from capture_backgrounds.py
│   └── bg_augmented_train.csv# (generated) hand samples composited onto those backgrounds
├── deploy/
│   ├── sync_model.sh         # Copies output/model.h into the ESP32 sketch folder
│   ├── esp32/
│   │   └── ASL_Detector/     # ESP32-S3 Sense sketch (camera + inference + car output)
│   └── arduino_mega/
│       └── ASL_Car_Controller/ # Arduino Mega sketch (receives letters, drives the car)
├── simulator/
│   └── index.html            # Web-based car simulator for local testing (no hardware)
└── output/                    # Generated models and debug artifacts appear here
    ├── asl_model.keras        # Re-trainable Keras model
    ├── asl_model.tflite       # Full int8 quantized microcontroller model
    ├── model.h                # C++ array for the ESP32
    └── debug_frames/          # (generated) frames saved with 's' during webcam testing
```

## 🚀 The Machine Learning Workflow

### Step 0: Grant Display Permissions (For Webcam Testing)

Because Docker is an isolated container, you must grant it permission to draw a window on your Ubuntu desktop. **You must run this command once every time you restart your computer.**

```bash
xhost +local:docker
```

### Step 1: Train (or Fine-Tune) the Model

This script reads `sign_mnist_train.csv`/`sign_mnist_valid.csv`, automatically merges in `custom_train.csv` and `bg_augmented_train.csv` if present (see Steps 1a/1b below), applies heavy visual augmentation (rotation, zoom, shear, brightness shifts) to simulate webcam conditions, and trains the CNN.

* **No `asl_model.keras` in `output/`:** builds a brand-new model from scratch and trains for 20 epochs.
* **`asl_model.keras` already exists:** loads it and fine-tunes at a much lower learning rate. Delete `output/asl_model.keras` first if you want a genuine from-scratch run instead of fine-tuning on top of an existing checkpoint.

```bash
docker compose run --rm pipeline python src/train_model.py
```

#### Step 1a: Collect Your Own Real-Hand Samples (recommended)

The base dataset is a narrow, studio-photographed dataset — it won't fully generalize to your specific webcam/lighting/hand. Collect your own labeled samples:

```bash
docker compose run --rm pipeline python src/capture_data.py
```

Two windows open: your live feed (with a guide box) and a preview of exactly what gets saved (same crop pipeline used at inference). Press the letter key matching the sign you're showing (A-I, K-Y — no J/Z, they require motion) to save a sample; press `q` to quit. Aim for ~15-20 samples per letter, varying angle/distance slightly each time. Samples accumulate in `data/custom_train.csv` across runs.

#### Step 1b: Make the Model Robust to Messy Backgrounds (optional)

If the model struggles whenever the background isn't plain (cluttered rooms, wood furniture, skin-toned objects confusing the hand crop), bake that robustness into the model itself rather than relying on perfect on-device segmentation (which isn't realistic on an ESP32S3's compute budget):

1. Capture some background-only photos (no hand in frame) of the messy environments you've seen problems in:

   ```bash
   docker compose run --rm pipeline python src/capture_backgrounds.py
   ```

   Press `s` to save a frame, `q` to quit. Move the camera around for variety. These go to `data/backgrounds/`.

2. Composite your `custom_train.csv` hand samples onto those backgrounds:

   ```bash
   docker compose run --rm pipeline python src/augment_backgrounds.py
   ```

   This segments the hand out of each sample (Otsu threshold) and pastes it onto random crops of your background photos, writing several variants per sample to `data/bg_augmented_train.csv`. Run with `--preview` first to sanity-check a few composites (written to `output/bg_preview/`) before committing to the full batch.

Then retrain (Step 1) — `train_model.py` automatically picks up both `custom_train.csv` and `bg_augmented_train.csv` if they exist.

### Step 2: Convert to TFLite + C++ (model.h)

Converts the trained Keras model to a **full int8 quantized** TFLite model (both weights and activations quantized, using a representative sample of the training data for calibration) — smaller and faster on the ESP32S3 than weights-only quantization, while keeping float32 input/output so `test_model.py` and the Arduino-side wrapper don't need to change.

```bash
docker compose run --rm pipeline python src/convert_model.py
```

*This generates `asl_model.tflite` and `model.h` in `output/`.*

### Step 3: Test with Live Webcam or Static Images

```bash
# Live webcam
docker compose run --rm pipeline python src/test_model.py webcam --keras

# Static images (data/a.png, data/b.png)
docker compose run --rm pipeline python src/test_model.py --keras

# Test the converted .tflite instead of the .keras model
docker compose run --rm pipeline python src/test_model.py webcam
```

`preprocess_image` auto-crops to the hand (skin-tone detection in YCrCb, then pads — not stretches — to a square) before resizing to 28x28, so the model gets a tightly-cropped input similar to the training data regardless of how much background the raw webcam frame captured.

In webcam mode, two windows appear: the live feed, and **"Model Input (28x28 preprocessed)"** showing exactly what the model sees — useful for sanity-checking the crop. Press `s` at any time to save the current raw ROI + model input to `output/debug_frames/` for later inspection; press `q` to quit.

### Step 3b: Run as a Stand-In for the ESP32 (`run_model.py`)

`run_model.py` shows the same live-webcam windows as `test_model.py`, but also mirrors what the ESP32-S3 Sense actually does on-device: it applies the same prediction smoothing (5-frame majority vote) and prints predictions to stdout in the exact same log format the firmware uses:

```text
[105167 ms] letter=A  confidence=98.8%
[106398 ms] ...
```

```bash
# TFLite model, logs to stdout only
docker compose run --rm pipeline python src/run_model.py webcam

# Keras model instead
docker compose run --rm pipeline python src/run_model.py webcam --keras
```

Useful on its own for testing the model/UI without touching hardware, and it doubles as the ESP32 replacement for [ASL Car Control](#-asl-car-control) below — see **Connection Option 3**.

---

## 🛠️ Deploying to the ESP32

When your model is accurately detecting your hand signs in Step 3, you are ready to flash the microcontroller.

1. Open your Arduino IDE.
2. Create a new sketch (e.g., `ASL_Detector.ino`).
3. Copy the `model.h` file from your `output/` folder into the same folder as your `.ino` sketch.
4. Install the **EloquentTinyML** library in Arduino.
5. In your Arduino code, include the model and initialize it using the exact variable name generated inside the header file:

```cpp
#include "model.h"
#include <EloquentTinyML.h>

// Initialize the TinyML wrapper using the array from model.h
Eloquent::TinyML::TfLite<784, 24, 120 * 1024> ml;

void setup() {
    ml.begin(_app_output_asl_model_tflite);
}
```

---

## 🚗 ASL Car Control

The camera recognizes ASL hand signs and sends them to an **Arduino Mega** that drives a car with dual motors and servo steering. Five ASL letters map to car actions:

| ASL Letter | Car Action | Description |
|---|---|---|
| **W** | Forward | Drive straight with encoder sync |
| **B** | Backward | Reverse straight |
| **C** | Left | Turn left while driving forward |
| **A** | Right | Turn right while driving forward |
| **O** | Stop | Stop all motors |

All other recognized letters are ignored. Only predictions with **≥ 80% confidence** are acted on. The car **auto-stops after 3 seconds** of no valid command (safety timeout).

### How it works

The ESP32-S3 Sense logs predictions over USB Serial at ~1.2-second intervals:

```text
[105167 ms] letter=A  confidence=98.8%
[106398 ms] letter=W  confidence=89.4%
[107629 ms] letter=W  confidence=92.1%
```

When a prediction passes the 80% confidence threshold, the ESP32 also sends just the letter character over a second serial line (`Serial1` on GPIO2) to the Arduino Mega, which executes the corresponding motor/servo command.

### Connection Option 1: Direct wire (recommended for a moving car)

Two wires between the boards — no computer needed once flashed:

```text
ESP32-S3 Sense          Arduino Mega
   D1 (GPIO2) ────TX────→ Pin 17 (RX2)
   GND        ───────────→ GND
```

> ⚠️ **Voltage levels:** The ESP32-S3 is 3.3V, the Arduino Mega is 5V. The TX→RX direction (3.3V → 5V) is safe — the Mega reads 3.3V as HIGH. If you ever need communication in the other direction, use a voltage divider.

**Steps:**

1. Sync the model into the ESP32 sketch (if you retrained):
   ```bash
   ./deploy/sync_model.sh
   ```

2. Flash the **ESP32-S3 Sense** — open `deploy/esp32/ASL_Detector/ASL_Detector.ino` in Arduino IDE:
   - Board: **XIAO_ESP32S3**
   - PSRAM: **OPI PSRAM**
   - USB CDC On Boot: **Enabled**
   - Library: **tflm_esp32**
   - Click **Upload**

3. Flash the **Arduino Mega** — open `deploy/arduino_mega/ASL_Car_Controller/ASL_Car_Controller.ino` in Arduino IDE:
   - Board: **Arduino Mega 2560**
   - Click **Upload**

4. Wire `D1 → Pin 17` and `GND → GND` between the boards.

5. Power both boards. Show an ASL sign to the camera, hold it steady for ~1–2 seconds — the car moves. Lower your hand — the car stops after 3 seconds.

### Connection Option 2: USB through a computer (for testing/debugging)

Both boards stay plugged into your computer via USB. A Python script reads the ESP32's log, filters by confidence, and forwards the letter to the Mega.

**Steps:**

1. Flash both boards the same way as Option 1.

2. Install the Python dependency:
   ```bash
   pip install pyserial
   ```

3. Find the COM ports — open Device Manager → Ports (COM & LPT). You'll see two ports, one per board.

4. Run the bridge:
   ```bash
   python src/serial_bridge.py --esp32 COM3 --car COM5
   ```
   *(Replace `COM3`/`COM5` with your actual port numbers.)*

   The bridge shows a live table of every prediction and whether it was forwarded or skipped:

   ```text
        TIME  LETTER     CONF  ACTION
      105167       D    98.8%  skip (low confidence)
      106398       W    89.4%  SEND → W
      107629       W    92.1%  SEND → W
   ```
   (The letter `D` is skipped because it's not a car command — only W/B/C/A/O are forwarded.)

5. To just monitor the camera without sending to the car:
   ```bash
   python src/serial_bridge.py --esp32 COM3 --dry-run
   ```

### Connection Option 3: No ESP32 — run the model on your computer

Don't have the ESP32-S3 flashed/wired yet, or want to iterate on the model without redeploying it? `src/run_model.py` runs inference on your laptop webcam and drives the Mega directly over USB, combining what the ESP32 firmware and `serial_bridge.py` do into one process — the Mega can't tell the difference.

**Steps:**

1. Flash just the **Arduino Mega** (see step 3 in Option 1) and plug it into your computer via USB.

2. Find its serial device:
   ```bash
   ls /dev/ttyACM* /dev/ttyUSB*
   ```

3. Expose that device to the Docker container — uncomment and adjust the path in `docker-compose.yml`:
   ```yaml
   devices:
     - /dev/video0:/dev/video0
     - /dev/ttyACM0:/dev/ttyACM0   # <- match what `ls` showed you
   ```
   Then rebuild once (`pyserial` was added to the image): `docker compose build`

4. Run it:
   ```bash
   docker compose run --rm pipeline python src/run_model.py webcam --keras --car /dev/ttyACM0
   ```
   *(Drop `--keras` to use the converted `.tflite` model instead; add `--threshold 80` to change the car's confidence gate from the 80% default.)*

   The live feed and model-input windows appear just like `test_model.py`/Step 3b, ESP32-format log lines print to the terminal, and whenever the smoothed prediction is one of W/B/C/A/O at ≥ 80% confidence, that letter is sent straight to the Mega — same effect as Option 2, no ESP32 required.

### Local simulator (no hardware needed)

Open `simulator/index.html` in any browser to test the full pipeline visually:

- **Keyboard mode** — press `W`, `C`, `L`, `R`, `S` to drive a virtual car. Press `X` to clear the trail.
- **Log Replay mode** — paste real ESP32 camera logs and click ▶ Replay. The simulator plays them back at real-time intervals, shows which predictions pass the 80% confidence threshold, and drives the car accordingly.
- **Live ESP32 mode** — with the ESP32-S3 plugged in over USB, click 🔌 Connect ESP32-S3 and pick its serial port. Uses the browser's [Web Serial API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Serial_API) (Chrome/Edge desktop only) to read the same `[ms] letter=X confidence=Y%` log lines the Arduino IDE Serial Monitor shows, and drives the virtual car from them in real time — no bridge script needed. Only one process can hold the serial port open at a time, so close the Arduino IDE's Serial Monitor (or `serial_bridge.py`) before connecting here.

The simulator includes a live timeout countdown bar, command feed with accept/reject status, and motor telemetry — everything the real car does, visualized.

---

## 🧠 Notes on Domain Shift

If the model works on the CSV data but struggles on your webcam, that's domain shift — the model has only ever seen the base dataset's studio-style photos. Steps 1a/1b above exist specifically to close that gap:

* **Wrong predictions everywhere, even on a plain background:** collect more real samples per letter via `capture_data.py`, focusing on whichever letters are misclassified.
* **Works on plain backgrounds but fails on cluttered/skin-toned ones:** run the background-augmentation workflow (Step 1b) so the model learns to ignore background clutter rather than relying on perfect segmentation.
