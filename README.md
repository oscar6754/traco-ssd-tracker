# TRACO 2024 HexBug Tracking

This project detects HexBug heads with a trained Torchvision SSD model, then links detections into stable identities over time.

## Main Files

- `dataset.py`: SSD training dataset for frame annotations.
- `train_ssd.py`: trains `ssd300_vgg16` with `num_classes=2`.
- `tracking/detection.py`: SSD loading, candidate filtering, box priors, NMS, optional motion filtering.
- `tracking/motion.py`: median background and motion-mask helpers.
- `tracking/multi_object_tracker.py`: tentative/confirmed multi-object tracker with Hungarian assignment.
- `predict_video.py`: end-to-end prediction for one video or a directory of videos.
- `evaluate_predictions.py`: local validation scoring with `get_score.py`.

The old prototype scripts are still present for reference, but `predict_video.py` is the clean inference path.

## Useful Commands

Regenerate SSD annotations after changing annotation code:

```powershell
.\.venv\Scripts\python.exe create_ssd_annotations.py
.\.venv\Scripts\python.exe split_annotations.py
```

Train the SSD model:

```powershell
.\.venv\Scripts\python.exe train_ssd.py
```

Predict one validation video:

```powershell
.\.venv\Scripts\python.exe predict_video.py --video training\training01.mp4 --output predictions_clean\training01.csv
```

Score that prediction:

```powershell
.\.venv\Scripts\python.exe evaluate_predictions.py --prediction predictions_clean\training01.csv --ground-truth training\training01.csv --log --make-video
```

Predict a folder of leaderboard videos:

```powershell
.\.venv\Scripts\python.exe predict_video.py --video-dir test --output-dir predictions_leaderboard
```

## CSV Format

Predictions are saved with the pandas index because `get_score.py` reads with `index_col=0`.

Required columns:

```csv
,t,hexbug,x,y
0,0,0,123.4,456.7
1,0,1,700.2,350.1
```

`hexbug` is a stable tracker ID, not a detector class. The pipeline keeps output IDs in the safe scorer range `0..10`.

## Data Notes

The training labels show up to 4 HexBugs per video. Most videos have 101 labeled frames, but a few tracks begin or end, and some labels go outside the frame. The new pipeline does not require a known HexBug count at inference time.
