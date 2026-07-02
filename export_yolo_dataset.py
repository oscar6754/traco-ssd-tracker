from pathlib import Path
import os
import shutil

import cv2
import pandas as pd


TRAIN_CSV = Path("annotations_train.csv")
VAL_CSV = Path("annotations_val.csv")
OUTPUT_DIR = Path("yolo_dataset")
CLASS_NAME = "hexbug_head"


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def clean_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def yolo_box(row, image_width, image_height):
    xmin = clamp(float(row["xmin"]), 0, image_width - 1)
    ymin = clamp(float(row["ymin"]), 0, image_height - 1)
    xmax = clamp(float(row["xmax"]), 0, image_width - 1)
    ymax = clamp(float(row["ymax"]), 0, image_height - 1)

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height
    return f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"


def write_split(annotations_csv, split_name):
    annotations = pd.read_csv(annotations_csv)
    image_dir = OUTPUT_DIR / "images" / split_name
    label_dir = OUTPUT_DIR / "labels" / split_name

    clean_directory(image_dir)
    clean_directory(label_dir)

    image_count = 0
    box_count = 0

    for image_path, rows in annotations.groupby("image_path"):
        source = Path(image_path)
        image = cv2.imread(str(source))
        if image is None:
            print(f"Skipping unreadable image: {source}")
            continue

        height, width = image.shape[:2]
        output_name = f"{source.parent.name}_{source.name}"
        output_image = image_dir / output_name
        output_label = label_dir / f"{Path(output_name).stem}.txt"

        labels = [
            label
            for _, row in rows.iterrows()
            if (label := yolo_box(row, width, height)) is not None
        ]
        if not labels:
            continue

        link_or_copy(source, output_image)
        output_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
        image_count += 1
        box_count += len(labels)

    print(f"{split_name}: {image_count} images, {box_count} boxes")


def write_data_yaml():
    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_DIR.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def clamp(value, low, high):
    return min(high, max(low, value))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_split(TRAIN_CSV, "train")
    write_split(VAL_CSV, "val")
    write_data_yaml()
    print(f"Saved YOLO dataset: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
