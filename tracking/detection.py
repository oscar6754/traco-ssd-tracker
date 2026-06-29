from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
from PIL import Image
from torchvision.models.detection import ssd300_vgg16
from torchvision.ops import nms
import torchvision.transforms.functional as F

from .motion import box_motion_ratio


@dataclass
class DetectorConfig:
    class_id: int = 1
    score_threshold: float = 0.25
    max_candidates: int = 25
    min_box_width: float = 12.0
    max_box_width: float = 120.0
    min_box_height: float = 12.0
    max_box_height: float = 120.0
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 2.50
    nms_iou_threshold: float = 0.20
    use_motion_filter: bool = True
    min_motion_ratio: float = 0.02
    motion_score_weight: float = 0.25


@dataclass
class DetectionCandidate:
    x: float
    y: float
    box: Tuple[float, float, float, float]
    score: float
    label: int
    motion: float = 0.0


def load_ssd_model(model_path, device):
    """Load the trained two-class SSD model."""
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )

    state_dict = torch.load(Path(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def _frame_to_tensor(frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return F.to_tensor(image).to(device)


def detect_candidates(
    model,
    frame_bgr,
    device,
    config,
    motion_mask: Optional[object] = None,
) -> List[DetectionCandidate]:
    """Run SSD and return cleaned head candidates for one frame."""
    image_tensor = _frame_to_tensor(frame_bgr, device)

    with torch.inference_mode():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()

    keep = labels == int(config.class_id)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    keep = scores >= float(config.score_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(boxes) == 0:
        return []

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    aspect_ratios = widths / torch.clamp(heights, min=1.0)

    keep = (
        (widths >= config.min_box_width)
        & (widths <= config.max_box_width)
        & (heights >= config.min_box_height)
        & (heights <= config.max_box_height)
        & (aspect_ratios >= config.min_aspect_ratio)
        & (aspect_ratios <= config.max_aspect_ratio)
    )

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(boxes) == 0:
        return []

    if motion_mask is None:
        motion_scores = torch.zeros((len(boxes),), dtype=torch.float32)
    else:
        motion_scores = torch.tensor(
            [box_motion_ratio(motion_mask, box.tolist()) for box in boxes],
            dtype=torch.float32,
        )

    if config.use_motion_filter and motion_mask is not None:
        keep = motion_scores >= float(config.min_motion_ratio)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        motion_scores = motion_scores[keep]

    if len(boxes) == 0:
        return []

    keep_indices = nms(
        boxes,
        scores,
        iou_threshold=float(config.nms_iou_threshold),
    )

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    motion_scores = motion_scores[keep_indices]

    ranking_scores = scores + float(config.motion_score_weight) * motion_scores

    if len(ranking_scores) > config.max_candidates:
        ranking = torch.argsort(ranking_scores, descending=True)[: config.max_candidates]
        boxes = boxes[ranking]
        scores = scores[ranking]
        labels = labels[ranking]
        motion_scores = motion_scores[ranking]
        ranking_scores = ranking_scores[ranking]

    order = torch.argsort(ranking_scores, descending=True)
    candidates = []

    for index in order.tolist():
        xmin, ymin, xmax, ymax = boxes[index].tolist()
        candidates.append(
            DetectionCandidate(
                x=(xmin + xmax) / 2.0,
                y=(ymin + ymax) / 2.0,
                box=(xmin, ymin, xmax, ymax),
                score=float(scores[index]),
                label=int(labels[index]),
                motion=float(motion_scores[index]),
            )
        )

    return candidates
