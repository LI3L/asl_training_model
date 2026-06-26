import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

print("Loading and preparing data...")
train_df = pd.read_csv("/app/data/sign_mnist_train.csv").fillna(0)
valid_df = pd.read_csv("/app/data/sign_mnist_valid.csv").fillna(0)

y_train = train_df["label"]
y_valid = valid_df["label"]
del train_df["label"]
del valid_df["label"]

x_train = train_df.values / 255
x_valid = valid_df.values / 255

num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

x_train = x_train.reshape(-1, 28, 28, 1)
x_valid = x_valid.reshape(-1, 28, 28, 1)

print("Building higher-capacity model...")
model = Sequential(
    [
        Input(shape=(28, 28, 1)),
        # Layer 1
        Conv2D(75, (3, 3), strides=1, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        # Layer 2
        Conv2D(100, (3, 3), strides=1, padding="same", activation="relu"),
        Dropout(0.2),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        # Layer 3
        Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"),
        Dropout(0.3),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        # Layer 4 (New layer to handle more complex webcam features)
        Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        Flatten(),
        Dense(units=512, activation="relu"),
        Dropout(0.4),  # Heavy dropout to prevent overfitting
        Dense(units=num_classes, activation="softmax"),
    ]
)

# ---------------------------------------------------------
# AGGRESSIVE AUGMENTATION: Simulating webcam conditions
# ---------------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=20,  # Increased: hands are rarely perfectly straight
    zoom_range=0.2,  # Increased: hands might be closer/further from camera
    width_shift_range=0.2,  # Increased: hands might not be perfectly centered
    height_shift_range=0.2,  # Increased
    brightness_range=[0.5, 1.5],  # NEW: simulates shadows and bright window light
    shear_range=0.15,  # NEW: simulates camera perspective distortion
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(x_train)

# Dynamically lower the learning rate if the model gets stuck
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.00001
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
batch_size = 32
img_iter = datagen.flow(x_train, y_train, batch_size=batch_size)

print("Training model...")
# Increased epochs to 20 because the harder dataset takes longer to learn
model.fit(
    img_iter,
    epochs=20,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_valid, y_valid),
    callbacks=[learning_rate_reduction],
)

model.save("/app/output/asl_model.keras")

print("Converting to Quantized TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

tflite_model_path = "/app/output/asl_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_quant_model)

print("Generating C++ Header File...")
header_path = "/app/output/model.h"
os.system(f"xxd -i {tflite_model_path} > {header_path}")

print("Success! Pipeline complete.")
