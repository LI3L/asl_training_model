// Runs the trained ASL model on the XIAO ESP32S3 Sense: grabs a grayscale
// frame from the onboard camera, feeds it to the on-device TFLite model,
// and logs every prediction (letter + confidence) over Serial.
//
// Deploy steps:
//   1. Run `../sync_model.sh` (or copy output/model.h here by hand) whenever
//      the model is retrained/reconverted.
//   2. In Arduino IDE -> Tools -> PSRAM -> select "OPI PSRAM" (required).
//   3. Install the "tflm_esp32" library in the Arduino IDE.
//   4. Select the "XIAO_ESP32S3" board and flash.
//   5. Open the Serial Monitor at 115200 baud to see the prediction log.

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

// Ops used by the trained CNN: QUANTIZE, CONV_2D, MUL, ADD,
// MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX, DEQUANTIZE.
#define TF_NUM_OPS 9

// Tensor arena in PSRAM (8 MB on the Sense board). Internal DRAM (~255 KB
// usable) is too small for this model's im2col scratch buffers. Requires
// Tools -> PSRAM -> OPI PSRAM to be set in Arduino IDE board settings.
#define ARENA_SIZE (512 * 1024)

MicroMutableOpResolver<TF_NUM_OPS> resolver;
MicroInterpreter *interpreter = nullptr;
uint8_t         *tensorArena  = nullptr;
TfLiteTensor    *modelInput   = nullptr;
TfLiteTensor    *modelOutput  = nullptr;

// Capture at 96x96 grayscale (smallest square frame the OV2640 supports) and
// downsample to the model's 28x28 input. Center your hand in frame and bring
// it close so it fills most of the view -- there is no on-device hand crop.
const int CAPTURE_SIZE = 96;

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

// Nearest-neighbor downsample of the 96x96 grayscale frame to 28x28,
// normalized to 0.0-1.0 (matches preprocess_image() in src/test_model.py).
void preprocess(const uint8_t *frame, float *out) {
  const int DST = 28;
  for (int y = 0; y < DST; y++) {
    int sy = y * CAPTURE_SIZE / DST;
    for (int x = 0; x < DST; x++) {
      int sx = x * CAPTURE_SIZE / DST;
      out[y * DST + x] = frame[sy * CAPTURE_SIZE + sx] / 255.0f;
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

  if (initCamera() != ESP_OK)
    halt("ERROR: camera init failed. Check camera_pins.h and board wiring.");

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

  Serial.printf("[%lu ms] letter=%c  confidence=%.1f%%\n",
      millis(), ALPHABET[best], modelOutput->data.f[best] * 100.0f);

  delay(200);
}
