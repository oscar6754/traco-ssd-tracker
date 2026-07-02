from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from helper import Hungarian


VIDEO_DIR = Path("test")
OUTPUT_DIR = Path("predictions")
MODEL_PATH = Path("best.pt")

IMAGE_SIZE = 1280
DEVICE = "auto"
SCORE_THRESHOLD = 0.04
IOU_THRESHOLD = 0.45
MAX_CANDIDATES = 40

MAX_TRACKS = 4
COUNT_SCORE_THRESHOLD = 0.10
COUNT_PERCENTILE = 80
ASSIGNMENT_DISTANCE = 280.0
SMOOTH_WINDOW = 1
EXTRA_TRACK_MIN_RATIO = 0.35
REACQUIRE_SCORE_THRESHOLD = 0.20
REACQUIRE_DISTANCE = 2200.0


@dataclass
class Candidate:
    x: float
    y: float
    score: float


def detect_candidates(model, frame):
    predict_args = {
        "source": frame,
        "imgsz": IMAGE_SIZE,
        "conf": SCORE_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "max_det": MAX_CANDIDATES,
        "verbose": False,
    }
    if DEVICE != "auto":
        predict_args["device"] = DEVICE

    result = model.predict(**predict_args)[0]
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()

    candidates = []
    for box, score, class_id in zip(boxes, scores, classes):
        if int(class_id) != 0:
            continue

        xmin, ymin, xmax, ymax = [float(value) for value in box]
        if xmax <= xmin or ymax <= ymin:
            continue

        candidates.append(
            Candidate(
                x=(xmin + xmax) / 2.0,
                y=(ymin + ymax) / 2.0,
                score=float(score),
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def read_video_detections(video_path, model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detections = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections.append(detect_candidates(model, frame))

    cap.release()
    return detections, width, height


def estimate_track_count(detections):
    counts = []

    for candidates in detections:
        count = sum(
            1
            for candidate in candidates
            if candidate.score >= COUNT_SCORE_THRESHOLD
        )
        count = min(count, MAX_TRACKS)
        if count > 0:
            counts.append(count)

    if not counts:
        return 0

    count = int(np.ceil(np.percentile(counts, COUNT_PERCENTILE)))
    return max(1, min(MAX_TRACKS, count))


def choose_start_frame(detections, track_count):
    best_frame = None
    best_score = -1.0

    for frame_index, candidates in enumerate(detections):
        if len(candidates) < track_count:
            continue

        score = sum(candidate.score for candidate in candidates[:track_count])
        if score > best_score:
            best_score = score
            best_frame = frame_index

    return best_frame


def make_track_states(seed_candidates, frame_index):
    return [
        {
            "x": candidate.x,
            "y": candidate.y,
            "vx": 0.0,
            "vy": 0.0,
            "last_frame": frame_index,
        }
        for candidate in seed_candidates
    ]


def predicted_position(state, frame_index):
    gap = max(1, abs(frame_index - state["last_frame"]))
    return state["x"] + state["vx"] * gap, state["y"] + state["vy"] * gap


def update_state(state, candidate, frame_index):
    gap = max(1, abs(frame_index - state["last_frame"]))
    new_vx = (candidate.x - state["x"]) / gap
    new_vy = (candidate.y - state["y"]) / gap

    state["vx"] = 0.6 * state["vx"] + 0.4 * new_vx
    state["vy"] = 0.6 * state["vy"] + 0.4 * new_vy
    state["x"] = candidate.x
    state["y"] = candidate.y
    state["last_frame"] = frame_index


def advance_without_detection(state, frame_index):
    state["x"], state["y"] = predicted_position(state, frame_index)
    state["last_frame"] = frame_index


def assign_candidates(states, candidates, frame_index):
    if not candidates:
        return {}

    costs = np.zeros((len(states), len(candidates)), dtype=float)
    distances = np.zeros_like(costs)

    for track_id, state in enumerate(states):
        pred_x, pred_y = predicted_position(state, frame_index)

        for candidate_id, candidate in enumerate(candidates):
            distance = float(np.hypot(pred_x - candidate.x, pred_y - candidate.y))
            distances[track_id, candidate_id] = distance
            costs[track_id, candidate_id] = distance + 30.0 * (1.0 - candidate.score)

    hungarian = Hungarian(costs)
    hungarian.calculate()

    assignments = {}
    assigned_candidates = set()
    for track_id, candidate_id in hungarian.get_results():
        if track_id >= len(states) or candidate_id >= len(candidates):
            continue
        if distances[track_id, candidate_id] <= ASSIGNMENT_DISTANCE:
            assignments[int(track_id)] = int(candidate_id)
            assigned_candidates.add(int(candidate_id))

    recover_lost_tracks(
        assignments,
        assigned_candidates,
        candidates,
        distances,
    )

    return assignments


def recover_lost_tracks(assignments, assigned_candidates, candidates, distances):
    free_tracks = [
        track_id
        for track_id in range(distances.shape[0])
        if track_id not in assignments
    ]
    free_candidates = [
        candidate_id
        for candidate_id, candidate in enumerate(candidates)
        if candidate_id not in assigned_candidates
        and candidate.score >= REACQUIRE_SCORE_THRESHOLD
    ]

    while free_tracks and free_candidates:
        best_pair = None
        best_distance = REACQUIRE_DISTANCE

        for track_id in free_tracks:
            for candidate_id in free_candidates:
                distance = float(distances[track_id, candidate_id])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (track_id, candidate_id)

        if best_pair is None:
            break

        track_id, candidate_id = best_pair
        assignments[int(track_id)] = int(candidate_id)
        free_tracks.remove(track_id)
        free_candidates.remove(candidate_id)


def follow_tracks(track_points, states, frame_numbers, detections):
    for frame_index in frame_numbers:
        assignments = assign_candidates(states, detections[frame_index], frame_index)

        for track_id, state in enumerate(states):
            candidate_id = assignments.get(track_id)
            if candidate_id is None:
                advance_without_detection(state, frame_index)
                continue

            candidate = detections[frame_index][candidate_id]
            update_state(state, candidate, frame_index)
            track_points[track_id][frame_index] = (candidate.x, candidate.y)


def build_tracks(detections, track_count):
    start_frame = choose_start_frame(detections, track_count)
    if start_frame is None:
        return []

    seed_candidates = detections[start_frame][:track_count]
    track_points = [
        {start_frame: (candidate.x, candidate.y)}
        for candidate in seed_candidates
    ]

    # The scorer punishes ID changes hard, so tracks are followed both ways
    # from a frame where YOLO is confident about all visible bugs.
    forward_states = make_track_states(seed_candidates, start_frame)
    follow_tracks(
        track_points,
        forward_states,
        range(start_frame + 1, len(detections)),
        detections,
    )

    backward_states = make_track_states(seed_candidates, start_frame)
    follow_tracks(
        track_points,
        backward_states,
        range(start_frame - 1, -1, -1),
        detections,
    )

    return track_points


def choose_tracks(detections):
    track_count = estimate_track_count(detections)
    if track_count == 0:
        return []

    track_points = build_tracks(detections, track_count)

    if track_count == 1:
        two_track_points = build_tracks(detections, 2)
        if has_stable_extra_track(two_track_points, len(detections)):
            return two_track_points

    return track_points


def has_stable_extra_track(track_points, total_frames):
    if len(track_points) < 2 or total_frames == 0:
        return False

    observed_ratios = [
        len(points) / total_frames
        for points in track_points
    ]
    return min(observed_ratios) >= EXTRA_TRACK_MIN_RATIO


def smooth_series(series):
    if SMOOTH_WINDOW <= 1:
        return series

    window = SMOOTH_WINDOW
    if window % 2 == 0:
        window += 1

    return series.rolling(window, center=True, min_periods=1).median()


def tracks_to_dataframe(track_points, total_frames, width, height):
    rows = []

    for track_id, points in enumerate(track_points):
        frame_range = range(total_frames)
        x_values = [points.get(frame, (None, None))[0] for frame in frame_range]
        y_values = [points.get(frame, (None, None))[1] for frame in frame_range]

        x_series = pd.Series(x_values, dtype=float).interpolate(limit_direction="both")
        y_series = pd.Series(y_values, dtype=float).interpolate(limit_direction="both")
        x_series = smooth_series(x_series)
        y_series = smooth_series(y_series)

        for frame_index in range(total_frames):
            rows.append(
                {
                    "t": frame_index,
                    "hexbug": track_id,
                    "x": clamp(float(x_series.iloc[frame_index]), 0, width - 1),
                    "y": clamp(float(y_series.iloc[frame_index]), 0, height - 1),
                }
            )

    return pd.DataFrame(rows, columns=["t", "hexbug", "x", "y"])


def clamp(value, low, high):
    return min(high, max(low, value))


def predict_video(video_path, output_csv, model):
    detections, width, height = read_video_detections(video_path, model)
    track_points = choose_tracks(detections)
    if not track_points:
        raise RuntimeError(f"No tracks found in {video_path}")

    prediction_df = tracks_to_dataframe(
        track_points=track_points,
        total_frames=len(detections),
        width=width,
        height=height,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_csv)
    print(f"{video_path.name}: {len(track_points)} tracks -> {output_csv}")


def main():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(str(MODEL_PATH))
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No videos found in {VIDEO_DIR}")

    for video_path in videos:
        output_csv = OUTPUT_DIR / f"{video_path.stem}.csv"
        predict_video(video_path, output_csv, model)


if __name__ == "__main__":
    main()
