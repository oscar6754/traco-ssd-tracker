from dataclasses import dataclass, replace
from pathlib import Path
import argparse

import cv2
import pandas as pd
import torch

from tracking import (
    DetectorConfig,
    MotionConfig,
    MultiObjectTracker,
    TrackerConfig,
    build_background_from_video,
    detect_candidates,
    load_ssd_model,
    make_motion_mask,
)


MODEL_PATH = Path("ssd_hexbug_best.pth")
OUTPUT_DIR = Path("predictions_clean")

DETECTOR_CONFIG = DetectorConfig(
    score_threshold=0.25,
    min_box_width=12.0,
    max_box_width=120.0,
    min_box_height=12.0,
    max_box_height=120.0,
    min_aspect_ratio=0.35,
    max_aspect_ratio=2.50,
    nms_iou_threshold=0.20,
    use_motion_filter=True,
    min_motion_ratio=0.02,
    max_candidates=25,
)

MOTION_CONFIG = MotionConfig(
    background_samples=60,
    pixel_threshold=30,
    open_kernel_size=5,
    dilate_iterations=2,
)

TRACKER_CONFIG = TrackerConfig(
    max_assignment_distance=220.0,
    velocity_alpha=0.60,
    max_missed_frames=12,
    max_tentative_missed=1,
    confirm_hits=3,
    new_track_min_distance=35.0,
    max_output_ids=11,
)


@dataclass
class OutputConfig:
    smooth_window: int = 5
    backfill_to_start_if_first_seen_before: int = 5
    ensure_final_frame: bool = True


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(name)


def read_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return info


def predict_video(
    video_path,
    output_csv,
    model,
    device,
    detector_config,
    motion_config,
    tracker_config,
    output_config,
    limit_frames=None,
):
    video_path = Path(video_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    info = read_video_info(video_path)
    print(f"Processing {video_path.name}: {info['width']}x{info['height']}")

    background = build_background_from_video(video_path, motion_config)
    tracker = MultiObjectTracker(tracker_config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    best_candidates_by_frame = {}
    frame_index = 0

    while True:
        if limit_frames is not None and frame_index >= limit_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        motion_mask = make_motion_mask(frame, background, motion_config)
        candidates = detect_candidates(
            model=model,
            frame_bgr=frame,
            device=device,
            config=detector_config,
            motion_mask=motion_mask,
        )

        if candidates:
            best_candidates_by_frame[frame_index] = candidates[0]

        tracker.update(candidates, frame_index)

        if frame_index % 25 == 0:
            print(
                f"  frame {frame_index:04d} | "
                f"candidates {len(candidates):02d} | "
                f"confirmed {len(tracker.confirmed_tracks()):02d}"
            )

        frame_index += 1

    cap.release()

    total_frames = frame_index
    prediction_df = tracks_to_prediction_df(
        tracks=tracker.confirmed_tracks(),
        total_frames=total_frames,
        width=info["width"],
        height=info["height"],
        output_config=output_config,
    )

    if prediction_df.empty:
        prediction_df = fallback_prediction_df(
            best_candidates_by_frame=best_candidates_by_frame,
            total_frames=total_frames,
            width=info["width"],
            height=info["height"],
            output_config=output_config,
        )

    if output_config.ensure_final_frame:
        prediction_df = ensure_frame_count(prediction_df, total_frames)

    if prediction_df.empty:
        raise RuntimeError(
            f"No confirmed tracks or fallback detections were produced for {video_path}."
        )

    prediction_df = prediction_df.sort_values(["t", "hexbug"]).reset_index(drop=True)

    # Keep the pandas index because get_score.py loads predictions with index_col=0.
    prediction_df.to_csv(output_csv)
    print(f"Saved {output_csv} ({len(prediction_df)} rows)")

    return output_csv


def tracks_to_prediction_df(tracks, total_frames, width, height, output_config):
    rows = []

    for track in tracks:
        frames = sorted(track.history)
        if not frames:
            continue

        first_frame = frames[0]
        last_frame = min(frames[-1], total_frames - 1)

        if first_frame <= output_config.backfill_to_start_if_first_seen_before:
            first_frame = 0

        x_by_frame = {
            frame: point.x
            for frame, point in track.history.items()
            if frame <= last_frame
        }
        y_by_frame = {
            frame: point.y
            for frame, point in track.history.items()
            if frame <= last_frame
        }

        track_rows = interpolate_track_rows(
            output_id=int(track.output_id),
            first_frame=first_frame,
            last_frame=last_frame,
            x_by_frame=x_by_frame,
            y_by_frame=y_by_frame,
            width=width,
            height=height,
            smooth_window=output_config.smooth_window,
        )
        rows.extend(track_rows)

    return pd.DataFrame(rows, columns=["t", "hexbug", "x", "y"])


def fallback_prediction_df(
    best_candidates_by_frame,
    total_frames,
    width,
    height,
    output_config,
):
    if not best_candidates_by_frame:
        return pd.DataFrame(columns=["t", "hexbug", "x", "y"])

    frames = sorted(best_candidates_by_frame)
    first_frame = frames[0]
    last_frame = min(frames[-1], total_frames - 1)

    if first_frame <= output_config.backfill_to_start_if_first_seen_before:
        first_frame = 0

    x_by_frame = {
        frame: candidate.x
        for frame, candidate in best_candidates_by_frame.items()
        if frame <= last_frame
    }
    y_by_frame = {
        frame: candidate.y
        for frame, candidate in best_candidates_by_frame.items()
        if frame <= last_frame
    }

    rows = interpolate_track_rows(
        output_id=0,
        first_frame=first_frame,
        last_frame=last_frame,
        x_by_frame=x_by_frame,
        y_by_frame=y_by_frame,
        width=width,
        height=height,
        smooth_window=output_config.smooth_window,
    )

    return pd.DataFrame(rows, columns=["t", "hexbug", "x", "y"])


def interpolate_track_rows(
    output_id,
    first_frame,
    last_frame,
    x_by_frame,
    y_by_frame,
    width,
    height,
    smooth_window,
):
    frame_range = list(range(int(first_frame), int(last_frame) + 1))

    x_values = [x_by_frame.get(frame) for frame in frame_range]
    y_values = [y_by_frame.get(frame) for frame in frame_range]

    x_series = pd.Series(x_values, dtype=float).interpolate(limit_direction="both")
    y_series = pd.Series(y_values, dtype=float).interpolate(limit_direction="both")

    if smooth_window > 1:
        window = int(smooth_window)
        if window % 2 == 0:
            window += 1

        x_series = x_series.rolling(window, center=True, min_periods=1).median()
        y_series = y_series.rolling(window, center=True, min_periods=1).median()

    rows = []

    for local_index, frame in enumerate(frame_range):
        x = clamp(float(x_series.iloc[local_index]), 0.0, float(width - 1))
        y = clamp(float(y_series.iloc[local_index]), 0.0, float(height - 1))
        rows.append(
            {
                "t": int(frame),
                "hexbug": int(output_id),
                "x": x,
                "y": y,
            }
        )

    return rows


def ensure_frame_count(prediction_df, total_frames):
    if prediction_df.empty or total_frames <= 0:
        return prediction_df

    max_t = int(prediction_df["t"].max())
    final_t = int(total_frames) - 1

    if max_t >= final_t:
        return prediction_df

    last_rows = prediction_df[prediction_df["t"] == max_t].copy()
    extra_rows = []

    for frame in range(max_t + 1, final_t + 1):
        for _, row in last_rows.iterrows():
            extra_rows.append(
                {
                    "t": int(frame),
                    "hexbug": int(row["hexbug"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )

    return pd.concat(
        [prediction_df, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )


def clamp(value, low, high):
    return max(low, min(high, value))


def collect_videos(args):
    if args.video is not None:
        return [Path(args.video)]

    video_dir = Path(args.video_dir)
    return sorted(video_dir.glob(args.pattern))


def output_path_for(video_path, args):
    if args.output is not None:
        return Path(args.output)

    output_dir = Path(args.output_dir)
    return output_dir / f"{Path(video_path).stem}.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRACO SSD detection and tracking.")

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path, help="Path to one .mp4 video.")
    inputs.add_argument("--video-dir", type=Path, help="Directory containing .mp4 videos.")

    parser.add_argument("--pattern", default="*.mp4", help="Video glob for --video-dir.")
    parser.add_argument("--output", type=Path, help="Output CSV for a single video.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for output CSV files.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to SSD .pth weights.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--limit-frames", type=int, help="Debug option: process only N frames.")

    parser.add_argument("--score-threshold", type=float, default=DETECTOR_CONFIG.score_threshold)
    parser.add_argument("--min-motion-ratio", type=float, default=DETECTOR_CONFIG.min_motion_ratio)
    parser.add_argument("--no-motion-filter", action="store_true")
    parser.add_argument("--motion-threshold", type=int, default=MOTION_CONFIG.pixel_threshold)
    parser.add_argument("--assignment-distance", type=float, default=TRACKER_CONFIG.max_assignment_distance)
    parser.add_argument("--confirm-hits", type=int, default=TRACKER_CONFIG.confirm_hits)
    parser.add_argument("--max-missed", type=int, default=TRACKER_CONFIG.max_missed_frames)
    parser.add_argument("--smooth-window", type=int, default=OutputConfig.smooth_window)

    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)

    detector_config = replace(
        DETECTOR_CONFIG,
        score_threshold=args.score_threshold,
        min_motion_ratio=args.min_motion_ratio,
        use_motion_filter=not args.no_motion_filter,
    )
    motion_config = replace(
        MOTION_CONFIG,
        pixel_threshold=args.motion_threshold,
    )
    tracker_config = replace(
        TRACKER_CONFIG,
        max_assignment_distance=args.assignment_distance,
        confirm_hits=args.confirm_hits,
        max_missed_frames=args.max_missed,
    )
    output_config = replace(
        OutputConfig(),
        smooth_window=args.smooth_window,
    )

    print(f"Using device: {device}")
    print(f"Loading model: {args.model}")
    model = load_ssd_model(args.model, device)

    videos = collect_videos(args)
    if not videos:
        raise RuntimeError("No videos matched the requested input.")

    for video_path in videos:
        predict_video(
            video_path=video_path,
            output_csv=output_path_for(video_path, args),
            model=model,
            device=device,
            detector_config=detector_config,
            motion_config=motion_config,
            tracker_config=tracker_config,
            output_config=output_config,
            limit_frames=args.limit_frames,
        )


if __name__ == "__main__":
    main()
