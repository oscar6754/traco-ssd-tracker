from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MotionConfig:
    background_samples: int = 60
    pixel_threshold: int = 30
    open_kernel_size: int = 5
    dilate_iterations: int = 2


def build_background_from_video(video_path, config):
    """Build a median background from sampled video frames."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count > 0:
        sample_count = min(config.background_samples, frame_count)
        frame_indices = np.linspace(0, frame_count - 1, sample_count).astype(int)
    else:
        frame_indices = np.arange(config.background_samples)

    frames = []

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()

        if ok and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"Could not sample frames from: {video_path}")

    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def make_motion_mask(frame, background, config):
    """Return a binary mask where white pixels are moving/changing areas."""
    if frame.shape[:2] != background.shape[:2]:
        raise ValueError("Frame and background have different sizes.")

    difference = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        gray,
        config.pixel_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel_size = max(1, int(config.open_kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if config.dilate_iterations > 0:
        mask = cv2.dilate(mask, kernel, iterations=config.dilate_iterations)

    return mask


def box_motion_ratio(mask, box):
    """Measure the fraction of motion pixels inside a detection box."""
    height, width = mask.shape[:2]
    xmin, ymin, xmax, ymax = [int(round(float(value))) for value in box]

    xmin = max(0, min(width - 1, xmin))
    xmax = max(0, min(width - 1, xmax))
    ymin = max(0, min(height - 1, ymin))
    ymax = max(0, min(height - 1, ymax))

    if xmax <= xmin or ymax <= ymin:
        return 0.0

    crop = mask[ymin:ymax, xmin:xmax]

    if crop.size == 0:
        return 0.0

    return float(crop.mean() / 255.0)
