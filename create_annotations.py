from pathlib import Path

import cv2
import pandas as pd


TRAINING_DIR = Path("training")
FRAMES_DIR = Path("frames")
OUTPUT_CSV = Path("annotations.csv")
BOX_RADIUS = 24


def clamp(value, low, high):
    return min(high, max(low, value))


def frame_path(frame_folder, frame_number):
    return frame_folder / f"frame_{int(frame_number):06d}.jpg"


def read_video_size(frame_folder, frame_numbers):
    for frame_number in frame_numbers:
        image = cv2.imread(str(frame_path(frame_folder, frame_number)))
        if image is not None:
            height, width = image.shape[:2]
            return width, height

    raise RuntimeError(f"Could not read any frame in {frame_folder}")


def box_around_point(x, y, width, height):
    xmin = clamp(x - BOX_RADIUS, 0, width - 1)
    ymin = clamp(y - BOX_RADIUS, 0, height - 1)
    xmax = clamp(x + BOX_RADIUS, 0, width - 1)
    ymax = clamp(y + BOX_RADIUS, 0, height - 1)
    return xmin, ymin, xmax, ymax


def main():
    rows = []
    skipped_boxes = 0
    csv_paths = sorted(TRAINING_DIR.glob("*.csv"))

    print(f"Creating annotations for {len(csv_paths)} videos...")

    for csv_path in csv_paths:
        video_name = csv_path.stem
        frame_folder = FRAMES_DIR / video_name

        if not frame_folder.exists():
            print(f"  missing frame folder: {frame_folder}")
            continue

        df = pd.read_csv(csv_path)
        width, height = read_video_size(frame_folder, df["t"].unique())

        for _, row in df.iterrows():
            t = int(row["t"])
            image_path = frame_path(frame_folder, t)

            if not image_path.exists():
                print(f"  missing frame: {image_path}")
                continue

            x = float(row["x"])
            y = float(row["y"])
            xmin, ymin, xmax, ymax = box_around_point(x, y, width, height)

            if xmax <= xmin or ymax <= ymin:
                skipped_boxes += 1
                continue

            rows.append(
                {
                    "image_path": str(image_path),
                    "video": video_name,
                    "t": t,
                    "hexbug": int(row["hexbug"]),
                    "x": x,
                    "y": y,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": 1,
                }
            )

    annotations = pd.DataFrame(rows)
    annotations.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Box radius: {BOX_RADIUS}")
    print(f"Total boxes: {len(annotations)}")
    print(f"Total images with labels: {annotations['image_path'].nunique()}")
    print(f"Skipped invalid boxes: {skipped_boxes}")


if __name__ == "__main__":
    main()
