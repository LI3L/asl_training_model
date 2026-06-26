import cv2
import numpy as np
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))
from test_model import preprocess_image, ALPHABET

OUTPUT_CSV = "/app/data/custom_train.csv"


def main():
    counts = {letter: 0 for letter in ALPHABET}
    file_exists = os.path.exists(OUTPUT_CSV)
    if file_exists:
        with open(OUTPUT_CSV) as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                counts[ALPHABET[int(row[0])]] += 1

    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(["label"] + [f"pixel{i}" for i in range(1, 785)])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open /dev/video0. Check permissions.")
        return

    print("Press a letter key (A-Y, no J/Z) to save a labeled sample of that sign.")
    print("Press 'q' to quit and write the session summary.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        cv2.rectangle(
            frame,
            (width // 2 - 150, height // 2 - 150),
            (width // 2 + 150, height // 2 + 150),
            (0, 255, 0),
            2,
        )
        roi = frame[
            height // 2 - 150 : height // 2 + 150, width // 2 - 150 : width // 2 + 150
        ]

        # Same crop+normalize pipeline used at inference, so what you
        # capture here matches what the model will actually be fed.
        input_data = preprocess_image(roi)
        preview = (input_data[0, :, :, 0] * 255).astype(np.uint8)
        preview_big = cv2.resize(preview, (280, 280), interpolation=cv2.INTER_NEAREST)

        total = sum(counts.values())
        cv2.putText(
            frame,
            f"Captured: {total}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture - Live Feed", frame)
        cv2.imshow("Capture - Model Input Preview", preview_big)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key != 255:
            letter = chr(key).upper()
            if letter in ALPHABET:
                label = ALPHABET.index(letter)
                writer.writerow([label] + preview.flatten().tolist())
                csv_file.flush()
                counts[letter] += 1
                print(f"Saved sample for '{letter}' (total for this letter: {counts[letter]})")
            elif letter in ("J", "Z"):
                print(f"'{letter}' requires motion and isn't in this dataset - skipped.")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nSession summary:")
    for letter in ALPHABET:
        print(f"  {letter}: {counts[letter]} samples")
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
