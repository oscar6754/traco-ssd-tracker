from pathlib import Path
import random

import pandas as pd


ANNOTATIONS_CSV = Path("annotations.csv")

TRAIN_CSV = Path("annotations_train.csv")
VAL_CSV = Path("annotations_val.csv")

TRAIN_RATIO = 0.80
SEED = 42


def main():
    df = pd.read_csv(ANNOTATIONS_CSV)
    videos = sorted(df["video"].unique())

    random.seed(SEED)
    random.shuffle(videos)

    train_end = int(len(videos) * TRAIN_RATIO)
    train_videos = videos[:train_end]
    val_videos = videos[train_end:]

    train_df = df[df["video"].isin(train_videos)]
    val_df = df[df["video"].isin(val_videos)]

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)

    print(f"Videos: {len(train_videos)} train, {len(val_videos)} validation")
    print(f"Images: {train_df['image_path'].nunique()} train, {val_df['image_path'].nunique()} validation")
    print(f"Saved: {TRAIN_CSV}, {VAL_CSV}")


if __name__ == "__main__":
    main()
