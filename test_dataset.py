from dataset import TracoSSDDataset


def main():
    dataset = TracoSSDDataset("ssd_annotations.csv")

    print("Number of images:", len(dataset))

    image, target = dataset[0]

    print("Image shape:", image.shape)
    print("Boxes:", target["boxes"])
    print("Labels:", target["labels"])


if __name__ == "__main__":
    main()
