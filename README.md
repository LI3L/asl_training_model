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
│   ├── capture_data.py       # Collect labeled real-hand samples from your webcam
│   ├── capture_backgrounds.py# Collect background-only photos (no hand) for augmentation
│   └── augment_backgrounds.py# Composite real-hand samples onto varied backgrounds
├── data/
│   ├── sign_mnist_train.csv  # Base Sign Language MNIST training set
│   ├── sign_mnist_valid.csv  # Base validation set
│   ├── custom_train.csv      # (generated) your real-hand samples from capture_data.py
│   ├── backgrounds/          # (generated) background-only photos from capture_backgrounds.py
│   └── bg_augmented_train.csv# (generated) hand samples composited onto those backgrounds
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

## 🧠 Notes on Domain Shift

If the model works on the CSV data but struggles on your webcam, that's domain shift — the model has only ever seen the base dataset's studio-style photos. Steps 1a/1b above exist specifically to close that gap:

* **Wrong predictions everywhere, even on a plain background:** collect more real samples per letter via `capture_data.py`, focusing on whichever letters are misclassified.
* **Works on plain backgrounds but fails on cluttered/skin-toned ones:** run the background-augmentation workflow (Step 1b) so the model learns to ignore background clutter rather than relying on perfect segmentation.
