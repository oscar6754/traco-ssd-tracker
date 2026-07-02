# TRACO 2024 YOLO Pipeline

This project detects and tracks HexBug heads for TRACO 2024.

The current pipeline is YOLO-only:

1. Build point-centered detection boxes.
2. Export annotations to YOLO format.
3. Train YOLO.
4. Predict videos with the simple YOLO tracker.
5. Score validation predictions with the provided scorer.

## Main Files

- `extract_frames.py`: extracts video frames into `frames/`.
- `create_annotations.py`: creates point-centered detection boxes with `BOX_RADIUS = 24`.
- `split_annotations.py`: creates train/validation annotation splits.
- `export_yolo_dataset.py`: writes the YOLO dataset under `yolo_dataset/`.
- `train_yolo.py`: trains or resumes a YOLO detector.
- `predict_video_yolo_simple.py`: final simple YOLO prediction pipeline.
- `evaluate_predictions.py`: scores predictions against local ground truth.
- `get_score.py` and `helper.py`: official/local scoring helpers.

## Build Data

```powershell
.\.venv\Scripts\python.exe create_annotations.py
.\.venv\Scripts\python.exe split_annotations.py
.\.venv\Scripts\python.exe export_yolo_dataset.py
```

## Train YOLO

The training values are fixed at the top of `train_yolo.py`.

```bash
python train_yolo.py
```

Resume after a time-limited GPU session:

```bash
python train_yolo.py --resume
```

Use `best.pt` for prediction. `last.pt` is only for continuing training.
Copy the final model to the project root as `best.pt`.

## Predict Test Videos

The current inference values are fixed at the top of `predict_video_yolo_simple.py`.

```bash
python predict_video_yolo_simple.py
```

The CSV files are written to `predictions/`.

## Score Local Predictions

```powershell
.\.venv\Scripts\python.exe evaluate_predictions.py --prediction-dir predictions_val --ground-truth-dir training --percent
```

Predictions are saved as CSV files with columns:

```csv
,t,hexbug,x,y
0,0,0,123.4,456.7
1,0,1,700.2,350.1
```
