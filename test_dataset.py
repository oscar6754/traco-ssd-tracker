from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F


class TracoSSDDataset(Dataset):
    def __init__(self, annotations_csv):
        self.annotations = pd.read_csv(annotations_csv)

        # One item = one image/frame that why unique
        self.image_paths = sorted(self.annotations["image_path"].unique())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        rows = self.annotations[self.annotations["image_path"] == image_path]

        boxes = rows[["xmin", "ymin", "xmax", "ymax"]].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Clamp boxes inside the image making sure inside image
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width - 1)   # xmin
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width - 1)   # xmax
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height - 1)  # ymin
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height - 1)  # ymax

        # Keep only valid boxes
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        image = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


dataset = TracoSSDDataset("ssd_annotations.csv")

print("Number of images:", len(dataset))

image, target = dataset[0]

print("Image shape:", image.shape)
print("Boxes:", target["boxes"])
print("Labels:", target["labels"])