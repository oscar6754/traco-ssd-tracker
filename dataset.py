from pathlib import Path
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class TracoSSDDataset(Dataset):
    """Torchvision SSD dataset for TRACO frame annotations."""

    REQUIRED_COLUMNS = {
        "image_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }

    def __init__(self, annotations_csv, augment=False):
        self.annotations_csv = Path(annotations_csv)
        self.augment = augment
        self.annotations = pd.read_csv(self.annotations_csv)

        missing = self.REQUIRED_COLUMNS - set(self.annotations.columns)
        if missing:
            raise ValueError(
                f"{self.annotations_csv} is missing columns: {sorted(missing)}"
            )

        self.image_paths = sorted(
            self.annotations["image_path"].dropna().astype(str).unique()
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size

            rows = self.annotations[self.annotations["image_path"] == image_path]
            boxes = self._get_boxes(rows, width, height)

            if self.augment:
                image, boxes = self._augment(image, boxes, width, height)

            image_tensor = F.to_tensor(image)

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        return image_tensor, {
            "boxes": boxes,
            "labels": labels,
        }

    def _get_boxes(self, rows, width, height):
        if rows.empty:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(
                rows[["xmin", "ymin", "xmax", "ymax"]].astype("float32").values,
                dtype=torch.float32,
            )

            boxes = self._clamp_boxes(boxes, width, height)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]

        return boxes

    @staticmethod
    def _clamp_boxes(boxes, width, height):
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width - 1)
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width - 1)
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height - 1)
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height - 1)
        return boxes

    def _augment(self, image, boxes, width, height):
        if random.random() < 0.50:
            image = F.hflip(image)
            boxes = boxes.clone()
            old_xmin = boxes[:, 0].clone()
            old_xmax = boxes[:, 2].clone()
            boxes[:, 0] = (width - 1) - old_xmax
            boxes[:, 2] = (width - 1) - old_xmin

        if random.random() < 0.50:
            image = F.vflip(image)
            boxes = boxes.clone()
            old_ymin = boxes[:, 1].clone()
            old_ymax = boxes[:, 3].clone()
            boxes[:, 1] = (height - 1) - old_ymax
            boxes[:, 3] = (height - 1) - old_ymin

        if random.random() < 0.80:
            image = F.adjust_brightness(image, random.uniform(0.70, 1.30))
            image = F.adjust_contrast(image, random.uniform(0.70, 1.30))
            image = F.adjust_saturation(image, random.uniform(0.75, 1.25))

        if random.random() < 0.15:
            image = F.gaussian_blur(image, kernel_size=3)

        return image, self._clamp_boxes(boxes, width, height)
