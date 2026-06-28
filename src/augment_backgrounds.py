import os
import csv
import sys
import cv2
import numpy as np

CUSTOM_CSV = "/app/data/custom_train.csv"
BG_DIR = "/app/data/backgrounds"
OUTPUT_CSV = "/app/data/bg_augmented_train.csv"
VARIANTS_PER_SAMPLE = 8


def load_backgrounds():
    paths = [
        os.path.join(BG_DIR, f)
        for f in os.listdir(BG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not paths:
        raise SystemExit(f"No background images found in {BG_DIR}. Run capture_backgrounds.py first.")
    backgrounds = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            backgrounds.append(img)
    return backgrounds


def hand_mask(img28):
    # Otsu split into two groups; assume whichever group dominates the
    # image border is background (dataset crops are tight, but a thin
    # border strip is almost always background, not hand).
    blurred = cv2.GaussianBlur(img28, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = 255 - mask
    # Clean up small noise specks from the threshold.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def random_bg_patch(bg, size=28):
    h, w = bg.shape
    if h < size or w < size:
        scale = size / min(h, w)
        bg = cv2.resize(bg, (int(w * scale) + 1, int(h * scale) + 1))
        h, w = bg.shape
    y0 = np.random.randint(0, h - size + 1)
    x0 = np.random.randint(0, w - size + 1)
    patch = bg[y0 : y0 + size, x0 : x0 + size]
    factor = np.random.uniform(0.7, 1.3)
    return np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def composite_row(pixels, mask, backgrounds):
    bg = backgrounds[np.random.randint(len(backgrounds))]
    patch = random_bg_patch(bg)
    return np.where(mask > 0, pixels, patch)


def main():
    preview_only = "--preview" in sys.argv

    backgrounds = load_backgrounds()
    print(f"Loaded {len(backgrounds)} background images.")

    with open(CUSTOM_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if preview_only:
        preview_dir = "/app/output/bg_preview"
        os.makedirs(preview_dir, exist_ok=True)
        sample_rows = rows[:: max(1, len(rows) // 6)][:6]
        for i, row in enumerate(sample_rows):
            pixels = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
            mask = hand_mask(pixels)
            composite = composite_row(pixels, mask, backgrounds)
            big = lambda im: cv2.resize(im, (280, 280), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{preview_dir}/{i}_original.png", big(pixels))
            cv2.imwrite(f"{preview_dir}/{i}_mask.png", big(mask))
            cv2.imwrite(f"{preview_dir}/{i}_composite.png", big(composite))
        print(f"Wrote preview triplets (original/mask/composite) to {preview_dir}")
        return

    print(f"Compositing {VARIANTS_PER_SAMPLE} background variants for each of {len(rows)} samples...")
    out_rows = []
    for row in rows:
        label = row[0]
        pixels = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
        mask = hand_mask(pixels)
        for _ in range(VARIANTS_PER_SAMPLE):
            composite = composite_row(pixels, mask, backgrounds)
            out_rows.append([label] + composite.flatten().tolist())

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} background-augmented samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
