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


def find_hand_bbox(image_bgr):
    # Skin-tone segmentation in YCrCb (more lighting-invariant than HSV/RGB)
    # to find the hand and crop tightly around it, matching the dataset's
    # tightly-cropped framing instead of whatever the raw ROI happened to capture.
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array((0, 135, 85), dtype=np.uint8)
    upper = np.array((255, 180, 135), dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None

    return cv2.boundingRect(largest)


def preprocess_image(image):
    # 1. Auto-crop to the hand if we can find one (webcam frames have a lot
    # of background; the dataset images are tightly cropped around the hand).
    if len(image.shape) == 3:
        bbox = find_hand_bbox(image)
        if bbox is not None:
            x, y, w, h = bbox
            height, width = image.shape[:2]
            pad_x, pad_y = int(w * 0.15), int(h * 0.15)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
            image = image[y0:y1, x0:x1]

            # Pad (don't stretch) the shorter side to square, replicating
            # the edge so the resize to 28x28 doesn't distort the hand's
            # proportions - the dataset images are square, undistorted crops.
            ch, cw = image.shape[:2]
            side = max(ch, cw)
            top = (side - ch) // 2
            bottom = side - ch - top
            left = (side - cw) // 2
            right = side - cw - left
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_REPLICATE
            )

    # 2. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 3. Resize to exactly 28x28 pixels
    resized = cv2.resize(gray, (28, 28))

    # 4. Normalize to 0.0 - 1.0 (Must match the float32 expectation)
    normalized = np.array(resized, dtype=np.float32) / 255.0

    # 5. Reshape to (1, 28, 28, 1) for the Keras Input layer
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
    print("Starting webcam... Press 'q' to quit, 's' to save the current frame for debugging.")
    save_dir = "/app/output/debug_frames"
    os.makedirs(save_dir, exist_ok=True)
    save_count = 0
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

        # Show exactly what the model sees: the 28x28 preprocessed crop,
        # scaled up so it's visible. Compare this to the dataset's look
        # (tightly cropped hand, plain background) to spot domain shift.
        debug_view = (input_data[0, :, :, 0] * 255).astype(np.uint8)
        debug_view = cv2.resize(
            debug_view, (280, 280), interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("Model Input (28x28 preprocessed)", debug_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"{save_dir}/roi_{save_count:03d}_pred_{letter}.png", roi)
            cv2.imwrite(
                f"{save_dir}/model_input_{save_count:03d}_pred_{letter}.png",
                debug_view,
            )
            print(f"Saved frame {save_count} (predicted {letter})")
            save_count += 1

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
