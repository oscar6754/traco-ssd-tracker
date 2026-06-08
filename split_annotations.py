import pandas as pd
import random

ANNOTATIONS_CSV = "ssd_annotations.csv"

TRAIN_CSV = "ssd_annotations_train.csv"
VAL_CSV = "ssd_annotations_val.csv"
TEST_CSV = "ssd_annotations_test.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

df = pd.read_csv(ANNOTATIONS_CSV)

videos = sorted(df["video"].unique())

random.seed(SEED)
random.shuffle(videos)

num_videos = len(videos)

train_end = int(num_videos * TRAIN_RATIO)
val_end = train_end + int(num_videos * VAL_RATIO)

train_videos = videos[:train_end]
val_videos = videos[train_end:val_end]
test_videos = videos[val_end:]

train_df = df[df["video"].isin(train_videos)]
val_df = df[df["video"].isin(val_videos)]
test_df = df[df["video"].isin(test_videos)]

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("Total videos:", num_videos)
print()

print("Train videos:", len(train_videos))
print(train_videos)
print("Train images:", train_df["image_path"].nunique())
print()

print("Validation videos:", len(val_videos))
print(val_videos)
print("Validation images:", val_df["image_path"].nunique())
print()

print("Test videos:", len(test_videos))
print(test_videos)
print("Test images:", test_df["image_path"].nunique())
print()

print("Saved:")
print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)