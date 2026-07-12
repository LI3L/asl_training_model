// Runs the trained ASL model on the XIAO ESP32S3 Sense: grabs a grayscale
// frame from the onboard camera, feeds it to the on-device TFLite model,
// logs every prediction (letter + confidence) over USB Serial, and -- for
// the five car-command letters (W/B/C/A/O) above CAR_CONFIDENCE_THRESHOLD --
// forwards just the letter over Serial1 (CAR_TX_PIN) to an Arduino Mega car
// controller. See the main README's "ASL Car Control" section.
//
// Deploy steps:
//   1. Run `../sync_model.sh` (or copy output/model.h here by hand) whenever
//      the model is retrained/reconverted.
//   2. In Arduino IDE -> Tools -> PSRAM -> select "OPI PSRAM" (required).
//   3. Install the "tflm_esp32" library in the Arduino IDE.
//   4. Select the "XIAO_ESP32S3" board and flash.
//   5. Open the Serial Monitor at 115200 baud to see the prediction log.
//   6. Optional: wire CAR_TX_PIN to the Mega's RX2 (pin 17) + shared GND to
//      drive a car -- see deploy/arduino_mega/ASL_Car_Controller.

#include "esp_camera.h"
#include "camera_pins.h"
#include "model.h"
#include "esp_heap_caps.h"
#include <tflm_esp32.h>

using tflite::MicroMutableOpResolver;
using tflite::MicroInterpreter;

// The ASL MNIST dataset maps labels 0-23 to these letters, skipping J and Z
// (they require motion) -- must match ALPHABET in src/test_model.py.
const char ALPHABET[] = "ABCDEFGHIKLMNOPQRSTUVWXY";

#define NUM_INPUTS  784   // 28 * 28
#define NUM_OUTPUTS  24   // len(ALPHABET)

// --- Car communication settings (see main README's "ASL Car Control" /
// "Connection Option 1: Direct wire" section) ---
// Minimum confidence (%, 0-100) to forward a stable letter to the car
// controller. Deliberately separate from MIN_CONFIDENCE below, which gates
// the on-device smoothing and is a 0.0-1.0 fraction, not a percentage.
#define CAR_CONFIDENCE_THRESHOLD 80.0f
// GPIO pin for Serial1 TX to the car (D1 on XIAO ESP32S3 Sense).
// Connect this pin to the Arduino Mega's RX2 (pin 17), and GND to GND.
#define CAR_TX_PIN 2

// Ops used by the trained CNN: QUANTIZE, CONV_2D, MUL, ADD,
// MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX, DEQUANTIZE.
#define TF_NUM_OPS 9

// Ignore single-frame guesses below this confidence entirely, and only log
// a letter once it's the majority of the last SMOOTH_WINDOW frames -- cuts
// down on flicker between noisy single-frame misreads.
#define MIN_CONFIDENCE 0.40f
#define SMOOTH_WINDOW 5
#define SMOOTH_MAJORITY 3

// Tensor arena in PSRAM (8 MB on the Sense board). Internal DRAM (~255 KB
// usable) is too small for this model's im2col scratch buffers. Requires
// Tools -> PSRAM -> OPI PSRAM to be set in Arduino IDE board settings.
#define ARENA_SIZE (512 * 1024)

MicroMutableOpResolver<TF_NUM_OPS> resolver;
MicroInterpreter *interpreter = nullptr;
uint8_t         *tensorArena  = nullptr;
TfLiteTensor    *modelInput   = nullptr;
TfLiteTensor    *modelOutput  = nullptr;

// Ring buffer of recent above-threshold predictions, used for majority-vote
// smoothing in loop() below.
int   smoothHistory[SMOOTH_WINDOW];
float smoothConf[SMOOTH_WINDOW];
int   smoothIdx = 0;

// Capture at 96x96 grayscale (smallest square frame the sensor supports).
// Center your hand in frame, filling most of the view, but stay outside the
// lens's close-focus blur zone -- roughly a hand's length (15-25cm) back,
// not pressed up against the camera.
const int CAPTURE_SIZE = 96;

// preprocess() below downsamples only the center CROP_FRACTION of the
// capture (discarding the outer margin) instead of squeezing the entire
// field of view into the model's 28x28 input. This is a fixed center crop,
// not real hand detection (too expensive for this board), but it trims
// unused background and makes the hand fill more of what the model
// actually sees, closer to the tightly-cropped training images.
const float CROP_FRACTION = 0.7f;

// Prints exactly what the model receives as ASCII art -- the on-device
// equivalent of the "Model Input (28x28 preprocessed)" debug window in
// src/test_model.py, which you don't get on a headless board. Use this to
// check the two most likely causes of "works on the desktop test, wrong on
// the device": (1) the hand isn't filling the frame the way it does in
// test_model.py's guide box, or (2) the sensor's image is mirrored/flipped
// relative to what the model was trained on -- if the printed hand shape
// looks left-right or upside-down flipped vs. what you're actually holding
// up, toggle FLIP_VERTICAL / MIRROR_HORIZONTAL below and re-flash.
#define DEBUG_PRINT_INPUT 0
const bool FLIP_VERTICAL = false;
const bool MIRROR_HORIZONTAL = false;

void printAsciiFrame(const float *img, int size) {
  static const char ramp[] = " .:-=+*#@";
  const int levels = sizeof(ramp) - 1;
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      int idx = (int)(img[y * size + x] * levels);
      if (idx < 0) idx = 0;
      if (idx >= levels) idx = levels - 1;
      Serial.print(ramp[idx]);
    }
    Serial.println();
  }
}

esp_err_t initCamera() {
  camera_config_t config = {};
  config.ledc_channel    = LEDC_CHANNEL_0;
  config.ledc_timer      = LEDC_TIMER_0;
  config.pin_d0          = Y2_GPIO_NUM;
  config.pin_d1          = Y3_GPIO_NUM;
  config.pin_d2          = Y4_GPIO_NUM;
  config.pin_d3          = Y5_GPIO_NUM;
  config.pin_d4          = Y6_GPIO_NUM;
  config.pin_d5          = Y7_GPIO_NUM;
  config.pin_d6          = Y8_GPIO_NUM;
  config.pin_d7          = Y9_GPIO_NUM;
  config.pin_xclk        = XCLK_GPIO_NUM;
  config.pin_pclk        = PCLK_GPIO_NUM;
  config.pin_vsync       = VSYNC_GPIO_NUM;
  config.pin_href        = HREF_GPIO_NUM;
  config.pin_sccb_sda    = SIOD_GPIO_NUM;
  config.pin_sccb_scl    = SIOC_GPIO_NUM;
  config.pin_pwdn        = PWDN_GPIO_NUM;
  config.pin_reset       = RESET_GPIO_NUM;
  config.xclk_freq_hz    = 20000000;
  config.pixel_format    = PIXFORMAT_GRAYSCALE;
  config.frame_size      = FRAMESIZE_96X96;
  config.fb_count        = 1;
  config.fb_location     = CAMERA_FB_IN_PSRAM;
  config.grab_mode       = CAMERA_GRAB_LATEST;
  return esp_camera_init(&config);
}

// Box-average downsample of the center CROP_FRACTION of the 96x96 grayscale
// frame to 28x28, normalized to 0.0-1.0 -- closer to cv2.resize()'s
// antialiased default than a nearest-neighbor pick, matching
// preprocess_image() in src/test_model.py more closely. Applies
// FLIP_VERTICAL / MIRROR_HORIZONTAL if set above.
void preprocess(const uint8_t *frame, float *out) {
  const int DST = 28;
  const int cropSize = (int)(CAPTURE_SIZE * CROP_FRACTION);
  const int cropOffset = (CAPTURE_SIZE - cropSize) / 2;

  for (int y = 0; y < DST; y++) {
    int sy0 = cropOffset + y * cropSize / DST;
    int sy1 = cropOffset + (y + 1) * cropSize / DST;
    if (sy1 <= sy0) sy1 = sy0 + 1;
    for (int x = 0; x < DST; x++) {
      int sx0 = cropOffset + x * cropSize / DST;
      int sx1 = cropOffset + (x + 1) * cropSize / DST;
      if (sx1 <= sx0) sx1 = sx0 + 1;

      uint32_t sum = 0;
      int count = 0;
      for (int sy = sy0; sy < sy1; sy++) {
        for (int sx = sx0; sx < sx1; sx++) {
          sum += frame[sy * CAPTURE_SIZE + sx];
          count++;
        }
      }

      int dx = MIRROR_HORIZONTAL ? (DST - 1 - x) : x;
      int dy = FLIP_VERTICAL ? (DST - 1 - y) : y;
      out[dy * DST + dx] = (sum / (float)count) / 255.0f;
    }
  }
}

void halt(const char *msg) {
  Serial.println(msg);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // Direct-wire link to the Arduino Mega car controller: TX only (RX
  // unused, hence -1) on CAR_TX_PIN, 9600 baud to match Serial2.begin(9600)
  // in ASL_Car_Controller.ino on the Mega side.
  Serial1.begin(9600, SERIAL_8N1, -1, CAR_TX_PIN);

  // -1 marks a slot as empty so it doesn't get counted as a vote for
  // letter A (index 0) before real predictions fill the window.
  for (int i = 0; i < SMOOTH_WINDOW; i++) {
    smoothHistory[i] = -1;
  }

  if (initCamera() != ESP_OK)
    halt("ERROR: camera init failed. Check camera_pins.h and board wiring.");

  // Max out contrast to help separate the hand from the background -- the
  // sensor's default settings otherwise produce a flat, low-contrast image
  // (visible as an ASCII debug frame using only mid-range characters with
  // no dark/bright ends, and no visible finger gaps).
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor) {
    Serial.printf("Camera sensor PID: 0x%02x\n", sensor->id.PID);
    sensor->set_contrast(sensor, 2);

    // OV3660 modules are commonly mounted rotated relative to OV2640 ones
    // on this board's camera header -- this is the standard correction
    // Espressif's own camera examples apply for that sensor specifically.
    if (sensor->id.PID == OV3660_PID) {
      sensor->set_vflip(sensor, 1);
      sensor->set_brightness(sensor, 1);
      sensor->set_saturation(sensor, -2);
    }
  }

  // Allocate tensor arena in PSRAM (Tools -> PSRAM -> OPI PSRAM must be set).
  tensorArena = (uint8_t *) heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM);
  if (!tensorArena)
    halt("ERROR: PSRAM arena allocation failed.\n"
         "In Arduino IDE: Tools -> PSRAM -> OPI PSRAM, then re-flash.");

  const tflite::Model *model = tflite::GetModel(_app_output_asl_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION)
    halt("ERROR: TFLite schema version mismatch -- reconvert the model.");

  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddDequantize();

  // Standard tflm_esp32 interpreter construction (no EloquentTinyML wrapper).
  interpreter = new MicroInterpreter(model, resolver, tensorArena, ARENA_SIZE);

  if (interpreter->AllocateTensors() != kTfLiteOk)
    halt("ERROR: AllocateTensors() failed -- try increasing ARENA_SIZE.");

  modelInput  = interpreter->input(0);
  modelOutput = interpreter->output(0);

  Serial.println("ASL detector ready. Center your hand in camera view.");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ERROR: frame capture failed");
    delay(200);
    return;
  }

  preprocess(fb->buf, modelInput->data.f);
  esp_camera_fb_return(fb);

#if DEBUG_PRINT_INPUT
  printAsciiFrame(modelInput->data.f, 28);
#endif

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed");
    delay(200);
    return;
  }

  int best = 0;
  for (int i = 1; i < NUM_OUTPUTS; i++) {
    if (modelOutput->data.f[i] > modelOutput->data.f[best])
      best = i;
  }
  float confidence = modelOutput->data.f[best];

  // Only feed above-threshold guesses into the smoothing window -- a
  // low-confidence frame shouldn't be able to win a majority vote.
  if (confidence >= MIN_CONFIDENCE) {
    smoothHistory[smoothIdx] = best;
    smoothConf[smoothIdx] = confidence;
    smoothIdx = (smoothIdx + 1) % SMOOTH_WINDOW;
  }

  int letterCounts[NUM_OUTPUTS] = {0};
  float letterConfSum[NUM_OUTPUTS] = {0};
  for (int i = 0; i < SMOOTH_WINDOW; i++) {
    int letter = smoothHistory[i];
    if (letter >= 0) {
      letterCounts[letter]++;
      letterConfSum[letter] += smoothConf[i];
    }
  }

  int stableLetter = -1, stableCount = 0;
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    if (letterCounts[i] > stableCount) {
      stableCount = letterCounts[i];
      stableLetter = i;
    }
  }

  if (stableLetter >= 0 && stableCount >= SMOOTH_MAJORITY) {
    char letter = ALPHABET[stableLetter];
    float avgConfidencePct = (letterConfSum[stableLetter] / stableCount) * 100.0f;
    Serial.printf("[%lu ms] letter=%c  confidence=%.1f%%\n",
        millis(), letter, avgConfidencePct);

    // Forward to the car over the direct wire (see main README's "ASL Car
    // Control" section) -- only the five car-command letters, and only
    // once they clear the stricter CAR_CONFIDENCE_THRESHOLD (independent
    // of the on-device smoothing's MIN_CONFIDENCE above). W/B/C/A/O map to
    // forward/backward/left/right/stop to match FORWARD/BACKWARD/LEFT/
    // RIGHT/STOP in the Mega's ASL_Car_Controller.ino.
    bool isCarCommand = (letter == 'W' || letter == 'B' || letter == 'C' ||
                          letter == 'A' || letter == 'O');
    if (isCarCommand && avgConfidencePct >= CAR_CONFIDENCE_THRESHOLD) {
      Serial1.write(letter);
      Serial1.write('\n');
    }
  } else {
    Serial.printf("[%lu ms] ...\n", millis());
  }

#if DEBUG_PRINT_INPUT
  delay(800);  // slower so the 28-line ASCII frame above stays readable
#else
  delay(20);   // just enough to yield to other tasks; camera+inference is the real limiter
#endif
}
