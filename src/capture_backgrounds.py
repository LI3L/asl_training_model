import cv2
import os

OUTPUT_DIR = "/app/data/backgrounds"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    count = len([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".png")])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open /dev/video0. Check permissions.")
        return

    print("Point the camera at messy/cluttered backgrounds with NO hand in frame.")
    print("Move/rotate the camera between shots to get variety. Press 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            f"Saved: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture Backgrounds (no hand in frame!)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"{OUTPUT_DIR}/bg_{count:03d}.png"
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total backgrounds saved: {count}")


if __name__ == "__main__":
    main()
