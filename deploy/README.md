# Deploying to the ESP32-S3 Sense

This folder contains the on-device inference code for the **Seeed Studio
XIAO ESP32S3 Sense**. It captures grayscale frames from the onboard camera,
runs them through the trained `model.h` (int8-quantized TFLite model, float32
in/out), and logs each prediction over Serial as:

```text
[12345 ms] letter=A confidence=97.32%
```

## Folder contents

```text
deploy/
├── sync_model.sh              # Copies output/model.h into the sketch folder
└── esp32/
    └── ASL_Detector/
        ├── ASL_Detector.ino   # Camera capture + inference + Serial logging
        ├── camera_pins.h      # XIAO ESP32S3 Sense camera pin mapping
        └── model.h            # Converted model (kept in sync via sync_model.sh)
```

## Arduino IDE setup, start to finish

### 1. Install the Arduino IDE

Download and install Arduino IDE 2.x from https://www.arduino.cc/en/software
for your OS, then launch it.

### 2. Add the ESP32 board package

1. Open **File > Preferences**.
2. In "Additional boards manager URLs", add:
   ```text
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Open **Tools > Board > Boards Manager**, search for **esp32** (by
   Espressif Systems), and install it.

### 3. Select and configure the board

1. Plug the XIAO ESP32S3 Sense into your computer with a **USB Type-C cable**
   (this is the only port on the board — it's used for both flashing and
   the Serial log).
2. Go to **Tools > Board > esp32 > XIAO_ESP32S3**.
3. Set the following under the **Tools** menu:
   - **USB CDC On Boot: Enabled** — routes `Serial` through the native USB
     peripheral so logs come out over the Type-C connector itself, instead
     of a separate UART pin. This is required for the logging setup below.
   - **PSRAM: OPI PSRAM** — the camera frame buffer needs it.
   - **Port**: select the serial port that appeared when you plugged in the
     board (e.g. `/dev/ttyACM0` on Linux, `COMx` on Windows).

### 4. Install the required library

Open **Tools > Manage Libraries**, search for **tflm_esp32**, and install
it. This is Espressif/EloquentArduino's Arduino port of TensorFlow Lite for
Microcontrollers — `ASL_Detector.ino` uses its `tflite::MicroInterpreter` /
`tflite::MicroMutableOpResolver` API directly (see "Why not EloquentTinyML"
below for why the higher-level wrapper isn't used here).

### 5. Sync the model and open the sketch

Whenever you retrain/reconvert the model, refresh the header the sketch uses:

```bash
./sync_model.sh
```

Then open `esp32/ASL_Detector/ASL_Detector.ino` in the Arduino IDE (it will
load `camera_pins.h` and `model.h` from the same folder automatically).

### 6. Upload

Click **Upload** (the right-arrow icon). The IDE compiles the sketch and
flashes it over the same Type-C cable.

### 7. Watch the prediction log

Open **Tools > Serial Monitor**, set the baud rate to **115200**, and make
sure it's connected to the same port selected in step 3. Predictions stream
in over the Type-C connector as:

```text
[12345 ms] letter=A confidence=97.32%
```

## If predictions are wrong on-device but correct in `test_model.py`

This means the model itself is fine — the problem is that what the ESP32
camera feeds it doesn't look like what `test_model.py` feeds it. Unlike
`test_model.py`'s "Model Input (28x28 preprocessed)" debug window, you can't
see what the ESP32 camera is capturing... unless you turn on the built-in
ASCII-art debug view.

`ASL_Detector.ino` has `#define DEBUG_PRINT_INPUT 1` near the top (on by
default). With it on, every prediction is preceded by a 28x28 ASCII-art
rendering of exactly what the model received. Open the Serial Monitor and
hold up a sign — you should see a rough hand shape rendered in `.:-=+*#@`
characters. Two things to check against it:

1. **Is the hand shape recognizable, with visible finger gaps, and does it
   fill most of the frame?** `test_model.py` uses a guide box + skin-tone
   crop to make the hand fill most of the model's input, matching the
   training data's tight framing — the ESP32 sketch has no equivalent crop
   (too expensive for its compute budget), so you have to do this manually:
   hold your hand roughly a hand's length back (~15-25cm), not pressed up
   against the lens. If the ASCII frame is a smooth, low-contrast blob using
   only mid-range characters (`-=+`) with no dark or bright ends and no
   visible gaps between fingers, that's the sensor's close-focus blur zone —
   back off until edges sharpen up.
2. **Is it mirrored or upside-down** compared to what you're actually
   holding up? Camera sensor mounting orientation varies, and a flipped
   input will confidently predict the wrong letter even though the model is
   working correctly. If so, set `FLIP_VERTICAL` and/or `MIRROR_HORIZONTAL`
   (also near the top of the file) to `true` and re-flash.

Once predictions look right, you can set `DEBUG_PRINT_INPUT` to `0` to go
back to a clean one-line-per-prediction log.

## Notes

- **Logs travel over the same Type-C connector used to flash the board.**
  The XIAO ESP32S3 has no separate USB-UART chip — its USB port connects
  directly to the ESP32-S3's native USB peripheral. With **USB CDC On Boot**
  enabled (step 3), `Serial` in `ASL_Detector.ino` is automatically backed by
  that native USB connection, so no extra wiring or hardware is needed to see
  the logs — just the same cable and the Serial Monitor.

- No on-device hand segmentation is attempted (too expensive for the
  ESP32S3's compute budget). `preprocess()` does apply a fixed center crop
  (`CROP_FRACTION`, default 0.7 — the center 70% of the 96x96 capture) before
  downsampling to 28x28, which trims some background and makes the hand fill
  more of the model's input, but it's not real hand detection — you still
  need to frame your hand centered and close-ish (see the distance note
  above). Robustness to backgrounds otherwise comes from training-time
  augmentation (see the main [README](../README.md#-notes-on-domain-shift)),
  not runtime cropping.
- Predictions are smoothed: a single frame's guess is only counted if its
  confidence is at least `MIN_CONFIDENCE` (default 40%), and a letter is
  only logged once it's the majority of the last `SMOOTH_WINDOW` frames
  (default 3 of the last 5). This trades a little latency for cutting down
  flicker between noisy single-frame misreads. While no letter has reached a
  majority yet, the log prints `[ms] ...` instead of a guess. Lower
  `SMOOTH_MAJORITY`/`SMOOTH_WINDOW` or `MIN_CONFIDENCE` for faster but
  noisier output, raise them for steadier but slower-to-update output.
- `ALPHABET` in `ASL_Detector.ino` must stay in sync with `ALPHABET` in
  `src/test_model.py` (both derive from the Sign Language MNIST label order,
  skipping J/Z since they require motion).
- `ARENA_SIZE` in `ASL_Detector.ino` is TFLite Micro's scratch memory for
  activations/im2col buffers, sized empirically against this specific
  architecture (retraining with a different architecture in
  `src/train_model.py` may need a different value). It's allocated with
  `heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM)` in PSRAM, not internal
  DRAM — internal DRAM only has ~255KB usable on this board once the rest of
  the firmware's globals are accounted for, which isn't enough for this
  model (a 120KB arena reported needing 273728 bytes for a single buffer
  resize during `AllocateTensors()`). If you see `ERROR: AllocateTensors()
  failed` or a `Failed to resize buffer` message on the Serial log, it
  prints `Requested/available/missing` byte counts — bump `ARENA_SIZE` by at
  least the `missing` amount and reflash; since it's in PSRAM (8MB on the
  Sense board) there's plenty of room to grow it.
- `sync_model.sh` patches the copied `model.h` to mark the model byte array
  `const` and 16-byte aligned. `xxd -i` (used by `src/convert_model.py`)
  emits a plain mutable array, which the linker places in the ESP32-S3's
  ~400KB of writable DRAM instead of leaving it memory-mapped in flash — a
  400KB+ model overflows that budget immediately (`DRAM segment data does
  not fit`). Always use `sync_model.sh` rather than copying `output/model.h`
  by hand, or you'll hit this.

#### Why not EloquentTinyML

An earlier version of this sketch used the higher-level `EloquentTinyML`
wrapper (`Eloquent::TF::Sequential`). Its tensor arena is a fixed-size array
*member of that class's global instance*, which the linker always places in
internal DRAM — there's no way to point it at PSRAM. That hits the ~255KB
DRAM ceiling described above before the model's actual arena requirement is
met (`DRAM segment data does not fit` / `region dram0_0_seg overflowed`).
Using the `tflm_esp32` library's lower-level API directly avoids the wrapper
so the arena can be PSRAM-allocated instead.
