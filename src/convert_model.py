import os
import tensorflow as tf
import sys

keras_model_path = "/app/output/asl_model.keras"
tflite_model_path = "/app/output/asl_model.tflite"
header_path = "/app/output/model.h"

if not os.path.exists(keras_model_path):
    print(f"Error: {keras_model_path} not found. You must run train_model.py first.")
    sys.exit(1)

print(f"Loading Keras model from {keras_model_path}...")
model = tf.keras.models.load_model(keras_model_path)

print("Converting to Quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

print(f"Saving TFLite model to {tflite_model_path}...")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"Generating C++ Header File at {header_path}...")
# Generate the C++ header file for the device
os.system(f"xxd -i {tflite_model_path} > {header_path}")

print("Success! Conversion complete. You can now use model.h or test the webcam.")
