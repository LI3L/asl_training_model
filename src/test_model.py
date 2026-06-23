import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# The ASL MNIST dataset maps 0-24 to A-Z, skipping J (9) and Z (25)
# because they require motion.
ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXY"


def load_tflite_model(model_path):
    print(f"Loading TFLite model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_keras_model(model_path):
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(image):
    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Resize to exactly 28x28 pixels
    resized = cv2.resize(gray, (28, 28))

    # 3. Normalize to 0.0 - 1.0 (Must match the float32 expectation)
    normalized = np.array(resized, dtype=np.float32) / 255.0

    # 4. Reshape to (1, 28, 28, 1) for the Keras Input layer
    input_data = np.expand_dims(normalized, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)

    return input_data


def predict(model, input_data, is_tflite):
    if is_tflite:
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Feed the data to the TFLite model
        model.set_tensor(input_details[0]["index"], input_data)
        model.invoke()

        # Read the output probabilities
        output_data = model.get_tensor(output_details[0]["index"])
    else:
        # Feed the data to the Keras model (verbose=0 stops console spam)
        output_data = model.predict(input_data, verbose=0)

    # Find the highest probability
    predicted_index = np.argmax(output_data[0])
    confidence = output_data[0][predicted_index]

    return ALPHABET[predicted_index], confidence


def test_static_images(model, is_tflite):
    # Put two test images in your /data folder
    image_paths = ["/app/data/a.png", "/app/data/b.png"]

    for path in image_paths:
        if not os.path.exists(path):
            print(
                f"Waiting for test image: {path} (Please add it to the 'data' folder)"
            )
            continue

        img = cv2.imread(path)
        input_data = preprocess_image(img)
        letter, confidence = predict(model, input_data, is_tflite)

        print(
            f"Image: {os.path.basename(path)} | Prediction: {letter} | Confidence: {confidence * 100:.2f}%"
        )


def test_webcam(model, is_tflite):
    print("Starting webcam... Press 'q' to quit.")
    # Initialize the default camera (0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open /dev/video0. Check permissions.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw a bounding box in the center to guide the user's hand
        height, width, _ = frame.shape
        cv2.rectangle(
            frame,
            (width // 2 - 150, height // 2 - 150),
            (width // 2 + 150, height // 2 + 150),
            (0, 255, 0),
            2,
        )

        # Crop the image to the bounding box for the model
        roi = frame[
            height // 2 - 150 : height // 2 + 150, width // 2 - 150 : width // 2 + 150
        ]

        # Run inference
        input_data = preprocess_image(roi)
        letter, confidence = predict(model, input_data, is_tflite)

        # Display the prediction on the video feed
        text = f"Pred: {letter} ({confidence * 100:.1f}%)"
        cv2.putText(
            frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3
        )

        cv2.imshow("ASL Live Inference", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Check for command line flags
    use_keras = "--keras" in sys.argv
    is_webcam = "webcam" in sys.argv

    if use_keras:
        model_path = "/app/output/asl_model.keras"
        if not os.path.exists(model_path):
            print(
                f"Error: Keras model not found at {model_path}. Run the training script first."
            )
            sys.exit(1)
        model = load_keras_model(model_path)
        is_tflite = False
    else:
        model_path = "/app/output/asl_model.tflite"
        if not os.path.exists(model_path):
            print(
                f"Error: TFLite model not found at {model_path}. Run the conversion script first."
            )
            sys.exit(1)
        model = load_tflite_model(model_path)
        is_tflite = True

    # Execute the requested test
    if is_webcam:
        test_webcam(model, is_tflite)
    else:
        test_static_images(model, is_tflite)
