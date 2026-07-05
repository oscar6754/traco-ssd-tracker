from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIDEO_DIR = Path("test")
PREDICTION_DIR = Path("predictions")
OUTPUT_DIR = Path("debug_videos")

# Use None for all videos, or write one name like "test005".
VIDEO_NAME = None

TRAIL_LENGTH = 40
POINT_RADIUS = 7
LINE_THICKNESS = 3

COLORS = [
    (0, 80, 255),
    (255, 80, 0),
    (0, 200, 0),
    (210, 0, 210),
    (0, 210, 210),
    (255, 160, 0),
]


def load_tracks(prediction_csv):
    df = pd.read_csv(prediction_csv, index_col=0)
    required_columns = {"t", "hexbug", "x", "y"}
    if not required_columns.issubset(df.columns):
        raise RuntimeError(f"{prediction_csv} does not contain {required_columns}")

    df = df.dropna(subset=["t", "hexbug", "x", "y"]).copy()
    df["t"] = df["t"].astype(int)
    df["hexbug"] = df["hexbug"].astype(int)

    tracks = {}
    for track_id, group in df.groupby("hexbug"):
        points = []
        for row in group.sort_values("t").itertuples():
            points.append((int(row.t), float(row.x), float(row.y)))
        tracks[int(track_id)] = points

    return tracks


def draw_tracks(frame, tracks, frame_index):
    height, width = frame.shape[:2]

    for track_id, points in tracks.items():
        color = COLORS[track_id % len(COLORS)]
        recent_points = [
            make_point(x, y, width, height)
            for t, x, y in points
            if frame_index - TRAIL_LENGTH <= t <= frame_index
        ]

        if len(recent_points) >= 2:
            cv2.polylines(
                frame,
                [np.array(recent_points, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=LINE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        current_point = get_point_at_frame(points, frame_index, width, height)
        if current_point is None:
            continue

        x, y = current_point
        cv2.circle(frame, (x, y), POINT_RADIUS + 4, (0, 0, 0), -1)
        cv2.circle(frame, (x, y), POINT_RADIUS, color, -1)
        cv2.drawMarker(
            frame,
            (x, y),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        draw_text(frame, str(track_id), (x + 12, y - 12), color)


def get_point_at_frame(points, frame_index, width, height):
    for t, x, y in points:
        if t == frame_index:
            return make_point(x, y, width, height)
    return None


def make_point(x, y, width, height):
    x = int(round(min(width - 1, max(0, x))))
    y = int(round(min(height - 1, max(0, y))))
    return x, y


def draw_text(frame, text, position, color):
    x, y = position
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_header(frame, video_name, frame_index, total_frames, track_count):
    text = f"{video_name}  frame {frame_index + 1}/{total_frames}  tracks {track_count}"
    cv2.rectangle(frame, (8, 8), (650, 48), (0, 0, 0), -1)
    cv2.putText(frame, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def write_debug_video(video_path, prediction_csv, output_path):
    tracks = load_tracks(prediction_csv)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_tracks(frame, tracks, frame_index)
        draw_header(frame, video_path.name, frame_index, total_frames, len(tracks))
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()


def iter_video_names():
    if VIDEO_NAME is not None:
        yield VIDEO_NAME
        return

    for prediction_csv in sorted(PREDICTION_DIR.glob("*.csv")):
        yield prediction_csv.stem


def main():
    for name in iter_video_names():
        video_path = VIDEO_DIR / f"{name}.mp4"
        prediction_csv = PREDICTION_DIR / f"{name}.csv"
        output_path = OUTPUT_DIR / f"{name}_debug.mp4"

        if not video_path.exists():
            print(f"Skipping {name}: missing {video_path}")
            continue
        if not prediction_csv.exists():
            print(f"Skipping {name}: missing {prediction_csv}")
            continue

        write_debug_video(video_path, prediction_csv, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
