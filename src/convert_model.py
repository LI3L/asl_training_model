import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

keras_model_path = "/app/output/asl_model.keras"
tflite_model_path = "/app/output/asl_model.tflite"
header_path = "/app/output/model.h"
train_csv_path = "/app/data/sign_mnist_train.csv"

if not os.path.exists(keras_model_path):
    print(f"Error: {keras_model_path} not found. You must run train_model.py first.")
    sys.exit(1)

print(f"Loading Keras model from {keras_model_path}...")
model = tf.keras.models.load_model(keras_model_path)

print(f"Loading representative dataset from {train_csv_path}...")
train_df = pd.read_csv(train_csv_path).fillna(0)
del train_df["label"]
x_train = train_df.values.reshape(-1, 28, 28, 1).astype("float32") / 255

# Sample a subset so the calibration pass stays fast; 500 images is plenty
# to estimate activation ranges for int8 quantization.
rng = np.random.default_rng(seed=42)
sample_indices = rng.choice(len(x_train), size=min(500, len(x_train)), replace=False)


def representative_dataset_gen():
    for i in sample_indices:
        yield [x_train[i : i + 1]]


print("Converting to full int8 Quantized TFLite (weights + activations)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Keep input/output as float32 so test_model.py and the EloquentTinyML
# wrapper can keep feeding/reading normalized floats unchanged.
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_quant_model = converter.convert()

print(f"Saving TFLite model to {tflite_model_path}...")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_quant_model)

print(f"Generating C++ Header File at {header_path}...")
# Generate the C++ header file for the device
os.system(f"xxd -i {tflite_model_path} > {header_path}")

print("Success! Conversion complete. You can now use model.h or test the webcam.")
