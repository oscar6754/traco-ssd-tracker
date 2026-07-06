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

# Low threshold on purpose
# The tracker can reject weak points, but cannot recover a missing bug
SCORE_THRESHOLD = 0.04
IOU_THRESHOLD = 0.45
MAX_CANDIDATES = 40
EDGE_MARGIN = 12

# Competition limit, still estimated from each video
MAX_TRACKS = 11
COUNT_SCORE_THRESHOLD = 0.10
COUNT_PERCENTILE = 80
DUPLICATE_DISTANCE = 30.0
ASSIGNMENT_DISTANCE = 280.0

# Track IDs start from a few stable frames, not just frame 0
START_FRAME_OPTIONS = 10
LOCAL_SUPPORT_FRAMES = 3
LOCAL_SUPPORT_DISTANCE = 120.0

MAX_REASONABLE_SPEED = 160.0
MAX_EXTRAPOLATION_SPEED = 120.0


@dataclass
class Detection:
    x: float
    y: float
    score: float


def detect_frame(model, frame):
    """YOLO detections converted to center points"""
    height, width = frame.shape[:2]
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

    detections = []
    for box, score, class_id in zip(boxes, scores, classes):
        if int(class_id) != 0:
            continue

        xmin, ymin, xmax, ymax = [float(value) for value in box]
        if xmax <= xmin or ymax <= ymin:
            continue

        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        if x < EDGE_MARGIN or y < EDGE_MARGIN:
            continue
        if x > width - EDGE_MARGIN or y > height - EDGE_MARGIN:
            continue

        detections.append(
            Detection(
                x=x,
                y=y,
                score=float(score),
            )
        )

    detections = sorted(detections, key=lambda detection: detection.score, reverse=True)
    return remove_duplicate_detections(detections)


def remove_duplicate_detections(detections):
    # Sometimes YOLO draws the same bug twice
    kept = []

    for detection in detections:
        is_duplicate = any(
            detection_distance(detection, other) < DUPLICATE_DISTANCE
            for other in kept
        )
        if not is_duplicate:
            kept.append(detection)

    return kept


def detection_distance(first, second):
    return float(np.hypot(first.x - second.x, first.y - second.y))


def read_video_detections(video_path, model):
    """Collect detections before tracking so we can use the full video"""
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

        detections.append(detect_frame(model, frame))

    cap.release()
    return detections, width, height


def estimate_track_count(detections):
    """Guess the fixed number of bugs in the video"""
    counts = []
    has_any_detection = False

    for frame_detections in detections:
        if frame_detections:
            has_any_detection = True

        count = sum(
            1
            for detection in frame_detections
            if detection.score >= COUNT_SCORE_THRESHOLD
        )
        count = min(count, MAX_TRACKS)
        if count > 0:
            counts.append(count)

    if not counts:
        return 1 if has_any_detection else 0

    count = int(np.ceil(np.percentile(counts, COUNT_PERCENTILE)))
    return max(1, min(MAX_TRACKS, count))


def choose_start_frames(detections, track_count):
    # Use the full video to avoid starting from a messy frame
    frame_scores = []

    for frame_index, frame_detections in enumerate(detections):
        if len(frame_detections) < track_count:
            continue

        score = score_start_frame(detections, frame_index, track_count)
        frame_scores.append((score, frame_index))

    frame_scores.sort(reverse=True)
    return [
        frame_index
        for _, frame_index in frame_scores[:START_FRAME_OPTIONS]
    ]


def score_start_frame(detections, frame_index, track_count):
    # Confidence plus nearby support gives a cleaner seed
    score = 0.0

    for detection in detections[frame_index][:track_count]:
        score += detection.score
        score += 0.05 * count_neighbor_support(detections, frame_index, detection)

    return score


def count_neighbor_support(detections, frame_index, detection):
    support = 0
    first_frame = max(0, frame_index - LOCAL_SUPPORT_FRAMES)
    last_frame = min(len(detections), frame_index + LOCAL_SUPPORT_FRAMES + 1)

    for neighbor_frame in range(first_frame, last_frame):
        if neighbor_frame == frame_index:
            continue

        has_nearby_detection = any(
            detection_distance(detection, other) <= LOCAL_SUPPORT_DISTANCE
            for other in detections[neighbor_frame]
        )
        if has_nearby_detection:
            support += 1

    return support


def make_track_states(seed_detections, frame_index):
    return [
        {
            "x": detection.x,
            "y": detection.y,
            "vx": 0.0,
            "vy": 0.0,
            "last_frame": frame_index,
        }
        for detection in seed_detections
    ]


def predicted_position(state, frame_index):
    # Short gaps are handled with a simple velocity guess
    gap = max(1, abs(frame_index - state["last_frame"]))
    return state["x"] + state["vx"] * gap, state["y"] + state["vy"] * gap


def update_state(state, detection, frame_index):
    gap = max(1, abs(frame_index - state["last_frame"]))
    new_vx = (detection.x - state["x"]) / gap
    new_vy = (detection.y - state["y"]) / gap

    state["vx"] = 0.6 * state["vx"] + 0.4 * new_vx
    state["vy"] = 0.6 * state["vy"] + 0.4 * new_vy
    state["x"] = detection.x
    state["y"] = detection.y
    state["last_frame"] = frame_index


def advance_without_detection(state, frame_index):
    # Keep moving through missed detections
    state["x"], state["y"] = predicted_position(state, frame_index)
    state["last_frame"] = frame_index


def assign_detections(states, frame_detections, frame_index):
    if not frame_detections:
        return {}

    # One detection per track, one track per detection
    costs = np.zeros((len(states), len(frame_detections)), dtype=float)
    distances = np.zeros_like(costs)

    for track_id, state in enumerate(states):
        pred_x, pred_y = predicted_position(state, frame_index)

        for detection_id, detection in enumerate(frame_detections):
            distance = float(np.hypot(pred_x - detection.x, pred_y - detection.y))
            distances[track_id, detection_id] = distance
            costs[track_id, detection_id] = distance + 30.0 * (1.0 - detection.score)

    hungarian = Hungarian(costs)
    hungarian.calculate()

    assignments = {}
    for track_id, detection_id in hungarian.get_results():
        if track_id >= len(states) or detection_id >= len(frame_detections):
            continue
        if distances[track_id, detection_id] <= ASSIGNMENT_DISTANCE:
            assignments[int(track_id)] = int(detection_id)

    return assignments


def follow_tracks(track_points, states, frame_numbers, detections):
    for frame_index in frame_numbers:
        assignments = assign_detections(states, detections[frame_index], frame_index)

        for track_id, state in enumerate(states):
            detection_id = assignments.get(track_id)
            if detection_id is None:
                advance_without_detection(state, frame_index)
                continue

            detection = detections[frame_index][detection_id]
            update_state(state, detection, frame_index)
            track_points[track_id][frame_index] = (detection.x, detection.y)


def build_tracks(detections, track_count, start_frame):
    seed_detections = detections[start_frame][:track_count]
    track_points = [
        {start_frame: (detection.x, detection.y)}
        for detection in seed_detections
    ]

    # Track forward and backward from the same seed frame
    # This reduces early-video ID mistakes
    forward_states = make_track_states(seed_detections, start_frame)
    follow_tracks(
        track_points,
        forward_states,
        range(start_frame + 1, len(detections)),
        detections,
    )

    backward_states = make_track_states(seed_detections, start_frame)
    follow_tracks(
        track_points,
        backward_states,
        range(start_frame - 1, -1, -1),
        detections,
    )

    return track_points


def choose_tracks(detections):
    # If the count was too high, try one less
    track_count = estimate_track_count(detections)
    while track_count > 0:
        best_tracks, _ = choose_best_tracks_for_count(detections, track_count)
        if best_tracks:
            return best_tracks

        track_count -= 1

    return []


def choose_best_tracks_for_count(detections, track_count):
    # Try several seeds and keep the smoothest set
    best_tracks = []
    best_score = -float("inf")

    for start_frame in choose_start_frames(detections, track_count):
        track_points = build_tracks(detections, track_count, start_frame)
        score = score_track_set(track_points)
        if score > best_score:
            best_score = score
            best_tracks = track_points

    return best_tracks, best_score


def score_track_set(track_points):
    # Long tracks are good; sudden jumps are suspicious
    score = 0.0

    for points in track_points:
        frames = sorted(points)
        score += len(frames)

        for previous_frame, next_frame in zip(frames, frames[1:]):
            frame_gap = next_frame - previous_frame
            if frame_gap <= 0:
                continue

            previous_x, previous_y = points[previous_frame]
            next_x, next_y = points[next_frame]
            speed = float(np.hypot(next_x - previous_x, next_y - previous_y)) / frame_gap
            if speed > MAX_REASONABLE_SPEED:
                score -= (speed - MAX_REASONABLE_SPEED) / 10.0

    return score


def tracks_to_dataframe(track_points, total_frames, width, height):
    rows = []

    for track_id, points in enumerate(track_points):
        x_series = fill_track_values(points, total_frames, value_index=0)
        y_series = fill_track_values(points, total_frames, value_index=1)

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


def fill_track_values(points, total_frames, value_index):
    # The CSV needs every frame, even when YOLO misses
    # Interpolate inside the track and extrapolate only at the edges
    values = np.full(total_frames, np.nan, dtype=float)

    for frame_index, point in points.items():
        values[frame_index] = point[value_index]

    known_frames = np.flatnonzero(~np.isnan(values))
    if len(known_frames) == 0:
        return pd.Series(values, dtype=float).fillna(0.0)
    if len(known_frames) == 1:
        return pd.Series(values, dtype=float).fillna(values[known_frames[0]])

    series = pd.Series(values, dtype=float).interpolate()
    extrapolate_start(series, values, known_frames)
    extrapolate_end(series, values, known_frames)
    return series


def extrapolate_start(series, values, known_frames):
    first_frame = known_frames[0]
    second_frame = known_frames[1]
    speed = bounded_speed(
        values[second_frame],
        values[first_frame],
        second_frame - first_frame,
    )

    for frame_index in range(first_frame):
        series.iloc[frame_index] = values[first_frame] - speed * (first_frame - frame_index)


def extrapolate_end(series, values, known_frames):
    previous_frame = known_frames[-2]
    last_frame = known_frames[-1]
    speed = bounded_speed(
        values[last_frame],
        values[previous_frame],
        last_frame - previous_frame,
    )

    for frame_index in range(last_frame + 1, len(series)):
        series.iloc[frame_index] = values[last_frame] + speed * (frame_index - last_frame)


def bounded_speed(new_value, old_value, frame_gap):
    speed = (new_value - old_value) / max(1, frame_gap)
    return clamp(speed, -MAX_EXTRAPOLATION_SPEED, MAX_EXTRAPOLATION_SPEED)


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
