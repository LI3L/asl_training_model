Here is a complete `README.md` file for your project. You can copy this and save it as `README.md` in the root of your `esp32-sense-pipeline` folder.

It covers the complete workflow, project structure, and all the Docker commands you will need.

---

# ESP32-S3 Sense: ASL TinyML Pipeline

This project is an automated Docker pipeline to train, fine-tune, and convert a custom American Sign Language (ASL) recognition model for the **Seeed Studio XIAO ESP32S3 Sense**.

It runs natively on Ubuntu Linux, leveraging your built-in webcam and X11 display server to test the model exactly as the microcontroller will see it, before compiling it into a C++ header file.

## 📁 Project Structure

Ensure your project directory looks exactly like this before running any commands:

```text
esp32-sense-pipeline/
├── docker-compose.yml       # Docker environment configuration
├── Dockerfile               # OS and dependency build instructions
├── src/
│   ├── train_model.py       # Trains or fine-tunes the .keras model
│   ├── convert_model.py     # Converts .keras -> .tflite -> model.h
│   └── test_model.py        # Live webcam OpenCV testing
├── data/                    # Put your training data here!
│   ├── sign_mnist_train.csv
│   └── sign_mnist_valid.csv
└── output/                  # Generated models appear here
    ├── asl_model.keras      # Re-trainable Keras model
    ├── asl_model.tflite     # Compiled microcontroller model
    └── model.h              # C++ array for the ESP32

```

## 🚀 The Machine Learning Workflow

This pipeline is split into three distinct steps. You run these commands from your Ubuntu terminal inside the `esp32-sense-pipeline` directory.

### Step 0: Grant Display Permissions (For Webcam Testing)

Because Docker is an isolated container, you must grant it permission to draw a window on your Ubuntu desktop. **You must run this command once every time you restart your computer.**

```bash
xhost +local:docker

```

### Step 1: Train (or Fine-Tune) the Model

This script reads your data, applies heavy visual augmentations (rotation, zoom, brightness shifts) to simulate webcam conditions, and trains the neural network.

* **First run:** It builds a brand new model and trains it for 20 epochs.
* **Subsequent runs:** It detects `asl_model.keras` in the `output/` folder, loads it, drops the learning rate, and fine-tunes it for 10 epochs.

```bash
docker compose run --rm pipeline python src/train_model.py

```

### Step 2: Convert to C++ (model.h)

Once you are happy with a training run, compile the model down to a highly compressed 8-bit quantized TensorFlow Lite model, and generate the raw C++ byte array.

```bash
docker compose run --rm pipeline python src/convert_model.py

```

*Note: This will generate `model.h` in your `output/` folder.*

### Step 3: Test with Live Webcam

This script connects to your laptop's `/dev/video0` camera, crops the image, converts it to 28x28 grayscale (exactly what the ESP32 does), and runs live inference.

```bash
docker compose run --rm pipeline python src/test_model.py webcam --keras

```

*(Press `q` on your keyboard while the window is focused to quit the camera).*

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

## 🧠 Pro-Tip: Fixing "Domain Shift"

If the model works on the CSV data but struggles on your webcam:
Take 20-50 pictures of your own hand doing signs in front of your actual webcam. Convert them to 28x28 grayscale images, add them to a folder in `data/`, and update `train_model.py` to include them. Fine-tuning the model on your actual room background will drastically improve the ESP32's accuracy!