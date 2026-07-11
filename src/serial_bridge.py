"""
serial_bridge.py — USB Bridge (Option 2)

Reads the ESP32-S3 Sense's prediction log from USB Serial, parses out the
letter and confidence, and forwards the letter to the Arduino Mega's USB
Serial whenever confidence >= 80%.

Usage:
    python serial_bridge.py --esp32 COM3 --car COM5

    --esp32   COM port of the ESP32-S3 Sense (115200 baud)
    --car     COM port of the Arduino Mega (9600 baud)
    --threshold  Minimum confidence to forward (default: 80.0)
    --dry-run    Don't open the car port; just print what would be sent

The ESP32 log format is:
    [12345 ms] letter=A  confidence=97.3%
"""

import argparse
import re
import sys
import time

try:
    import serial
except ImportError:
    print("ERROR: pyserial is not installed. Run:")
    print("  pip install pyserial")
    sys.exit(1)

# Regex to parse the ESP32 log line
LOG_PATTERN = re.compile(
    r"\[(\d+)\s*ms\]\s*letter=([A-Z])\s+confidence=([\d.]+)%"
)


def parse_log_line(line):
    """Parse an ESP32 log line. Returns (timestamp_ms, letter, confidence) or None."""
    match = LOG_PATTERN.search(line)
    if match:
        timestamp = int(match.group(1))
        letter = match.group(2)
        confidence = float(match.group(3))
        return timestamp, letter, confidence
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Bridge ESP32 ASL predictions to Arduino car controller"
    )
    parser.add_argument(
        "--esp32", required=True,
        help="COM port for ESP32-S3 Sense (e.g., COM3 or /dev/ttyACM0)"
    )
    parser.add_argument(
        "--car", default=None,
        help="COM port for Arduino Mega (e.g., COM5 or /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--threshold", type=float, default=80.0,
        help="Minimum confidence %% to forward a letter (default: 80.0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't open the car port; just print what would be sent"
    )
    args = parser.parse_args()

    # Open ESP32 serial
    print(f"Opening ESP32 on {args.esp32} at 115200 baud...")
    esp32 = serial.Serial(args.esp32, 115200, timeout=1)
    print("  Connected.")

    # Open car serial (if not dry-run)
    car = None
    if not args.dry_run and args.car:
        print(f"Opening Arduino car on {args.car} at 9600 baud...")
        car = serial.Serial(args.car, 9600, timeout=1)
        print("  Connected.")
        # Wait for Arduino to reset after serial connection
        time.sleep(2)
    elif args.dry_run:
        print("Dry-run mode: will not send to car.")
    else:
        print("WARNING: No --car port specified. Running in monitor-only mode.")

    print(f"\nBridge active. Threshold = {args.threshold}%")
    print("=" * 60)
    print(f"{'TIME':>10}  {'LETTER':>6}  {'CONF':>7}  {'ACTION'}")
    print("-" * 60)

    try:
        while True:
            # Read a line from ESP32
            raw = esp32.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            parsed = parse_log_line(line)
            if parsed is None:
                # Not a prediction line; print as-is (e.g., startup messages)
                print(f"  [ESP32] {line}")
                continue

            timestamp, letter, confidence = parsed

            # Check against threshold
            if confidence >= args.threshold:
                action = f"SEND → {letter}"
                if car:
                    car.write(f"{letter}\n".encode())
                elif args.dry_run:
                    action = f"(dry) → {letter}"
            else:
                action = "skip (low confidence)"

            print(f"{timestamp:>10}  {letter:>6}  {confidence:>6.1f}%  {action}")

    except KeyboardInterrupt:
        print("\n\nBridge stopped by user.")
    finally:
        esp32.close()
        if car:
            car.close()


if __name__ == "__main__":
    main()
