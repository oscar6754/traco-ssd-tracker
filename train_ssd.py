import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights

from test_dataset import TracoSSDDataset


TRAIN_CSV = "ssd_annotations_train.csv"
VAL_CSV = "ssd_annotations_val.csv"

NUM_CLASSES = 2  # background + hexbug_head

BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

SAVE_EVERY_EPOCH = True
BEST_MODEL_PATH = "ssd_hexbug_best.pth"


def collate_fn(batch):
    return tuple(zip(*batch))


def move_targets_to_device(targets, device):
    return [
        {
            "boxes": target["boxes"].to(device),
            "labels": target["labels"].to(device),
        }
        for target in targets
    ]


def evaluate_loss(model, val_loader, device, use_amp):
  
    model.train() #return losses

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): # this is for validation so no backpropagation, this faster
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = move_targets_to_device(targets, device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print("Non-finite validation loss detected, skipping batch.")
                continue

            total_loss += losses.item()
            num_batches += 1

    if num_batches == 0:
        return float("nan")

    return total_loss / num_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Using AMP:", use_amp)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = TracoSSDDataset(TRAIN_CSV)
    val_dataset = TracoSSDDataset(VAL_CSV)

    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=NUM_CLASSES,
    )

    model.to(device)

    # for update model weights
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    #reduce learning rate if validation loss stops going better
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_train_loss = 0.0
        num_train_batches = 0

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = move_targets_to_device(targets, device)

            # just clean previous ones
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print("Non-finite training loss detected, skipping batch.")
                print(loss_dict)
                continue

            #backpropagation
            scaler.scale(losses).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += losses.item()
            num_train_batches += 1

            #printing to get some info
            ''''''
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Train loss: {losses.item():.4f} "
                    f"LR: {current_lr:.2e}"
                )

        avg_train_loss = total_train_loss / max(num_train_batches, 1)

        val_loss = evaluate_loss(
            model=model,
            val_loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1} train loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")

        if SAVE_EVERY_EPOCH:
            epoch_model_path = f"ssd_hexbug_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Saved epoch model: {epoch_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model: {BEST_MODEL_PATH}")

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved as: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()