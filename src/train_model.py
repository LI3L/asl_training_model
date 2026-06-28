import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

print("Loading and preparing data...")
train_df = pd.read_csv("/app/data/sign_mnist_train.csv").fillna(0)
valid_df = pd.read_csv("/app/data/sign_mnist_valid.csv").fillna(0)

custom_path = "/app/data/custom_train.csv"
if os.path.exists(custom_path):
    custom_df = pd.read_csv(custom_path).fillna(0)
    print(
        f"Found {len(custom_df)} custom webcam samples in {custom_path}, merging into training data..."
    )
    train_df = pd.concat([train_df, custom_df], ignore_index=True)

bg_augmented_path = "/app/data/bg_augmented_train.csv"
if os.path.exists(bg_augmented_path):
    bg_df = pd.read_csv(bg_augmented_path).fillna(0)
    print(
        f"Found {len(bg_df)} background-augmented samples in {bg_augmented_path}, merging into training data..."
    )
    train_df = pd.concat([train_df, bg_df], ignore_index=True)

y_train = train_df["label"]
y_valid = valid_df["label"]
del train_df["label"]
del valid_df["label"]

num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Keep raw 0-255 pixel values here; ImageDataGenerator's rescale=1/255 below
# normalizes AFTER augmentation. Normalizing first breaks brightness_range,
# which expects 0-255 input and otherwise crushes every image to all-zero.
x_train = train_df.values.reshape(-1, 28, 28, 1).astype("float32")
x_valid = valid_df.values.reshape(-1, 28, 28, 1).astype("float32") / 255

keras_model_path = "/app/output/asl_model.keras"

# Check if model already exists to fine-tune it
if os.path.exists(keras_model_path):
    print(f"Found existing model! Loading {keras_model_path} for fine-tuning...")
    model = load_model(keras_model_path)
    # Recompile with a very low learning rate for fine-tuning
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
else:
    print("No existing model found. Building from scratch...")
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(75, (3, 3), strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(100, (3, 3), strides=1, padding="same", activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"),
            Dropout(0.3),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Flatten(),
            Dense(units=512, activation="relu"),
            Dropout(0.4),
            Dense(units=num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

# Aggressive webcam augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=35,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.4, 1.6],
    shear_range=0.25,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(x_train)

lr_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.00001
)

batch_size = 32
img_iter = datagen.flow(x_train, y_train, batch_size=batch_size)

epochs = 20
print(f"Training model for {epochs} epochs...")
model.fit(
    img_iter,
    epochs=epochs,
    steps_per_epoch=len(x_train) // batch_size,
    validation_data=(x_valid, y_valid),
    callbacks=[lr_reduction],
)

print(f"Saving Keras model to {keras_model_path}...")
model.save(keras_model_path)
print("Training phase complete!")
