from pathlib import Path
import argparse


DATA_YAML = Path("yolo_dataset/data.yaml")
RUN_NAME = "traco_yolo_m_r24_1280_4gpu"
START_MODEL = "yolo11m.pt"
LAST_MODEL = Path("runs/detect") / RUN_NAME / "weights" / "last.pt"

IMAGE_SIZE = 1280
BATCH_SIZE = 8
EPOCHS = 50
TIME_LIMIT_HOURS = 3.7
DEVICE = "0,1,2,3"

MOSAIC = 0.30
SCALE = 0.25
TRANSLATE = 0.05
DEGREES = 5.0
CLOSE_MOSAIC = 10
PATIENCE = 25


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TRACO YOLO model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from runs/detect/.../weights/last.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc

    if args.resume:
        model = YOLO(str(LAST_MODEL))
        model.train(
            resume=True,
            time=TIME_LIMIT_HOURS,
            device=DEVICE,
        )
        return

    model = YOLO(START_MODEL)
    model.train(
        data=str(DATA_YAML),
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        time=TIME_LIMIT_HOURS,
        name=RUN_NAME,
        device=DEVICE,
        patience=PATIENCE,
        mosaic=MOSAIC,
        scale=SCALE,
        translate=TRANSLATE,
        degrees=DEGREES,
        close_mosaic=CLOSE_MOSAIC,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
