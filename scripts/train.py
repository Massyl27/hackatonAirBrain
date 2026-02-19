#!/usr/bin/env python3
"""
Training script for PointNet-lite segmentation.
Optimized for CPU with multi-threaded DataLoader.

Usage:
    python train.py --data-dir /path/to/data --output-dir /path/to/output [--epochs 50]
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import LidarSegDataset
from model import PointNetLiteSeg, count_parameters

# ============================================================================
# CONFIG
# ============================================================================

SCENE_FILES = [f"scene_{i}.h5" for i in range(1, 11)]
VAL_SCENE = "scene_8"
NUM_CLASSES = 5
CLASS_NAMES = {0: "background", 1: "antenna", 2: "cable", 3: "electric_pole", 4: "wind_turbine"}

# Class weights from Story 1.2 analysis
CLASS_WEIGHTS = [0.1, 14.14, 70.41, 49.94, 6.29]


def get_train_val_indices(dataset, val_scene="scene_8"):
    """Split dataset into train/val by scene name."""
    train_idx, val_idx = [], []
    for i, entry in enumerate(dataset.index):
        scene_name = entry[7]  # scene_name field
        if scene_name == val_scene:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx


def compute_metrics(preds, labels, num_classes=5):
    """Compute per-class IoU and accuracy."""
    metrics = {}
    correct = (preds == labels).sum().item()
    total = labels.numel()
    metrics["accuracy"] = correct / total

    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
            metrics[f"iou_{CLASS_NAMES[c]}"] = intersection / union
        else:
            metrics[f"iou_{CLASS_NAMES[c]}"] = float("nan")

    metrics["mean_iou"] = np.nanmean(ious) if ious else 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_points = 0

    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)  # (B, N, 4)
        labels = labels.to(device)      # (B, N)

        optimizer.zero_grad()
        logits = model(features)         # (B, N, 5)
        logits_flat = logits.reshape(-1, NUM_CLASSES)
        labels_flat = labels.reshape(-1)

        loss = criterion(logits_flat, labels_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = logits_flat.argmax(dim=1)
        total_correct += (preds == labels_flat).sum().item()
        total_points += labels_flat.numel()

    return total_loss / len(loader.dataset), total_correct / total_points


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        logits_flat = logits.reshape(-1, NUM_CLASSES)
        labels_flat = labels.reshape(-1)

        loss = criterion(logits_flat, labels_flat)
        total_loss += loss.item() * features.size(0)

        all_preds.append(logits_flat.argmax(dim=1).cpu())
        all_labels.append(labels_flat.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PointNet-lite segmentation")
    parser.add_argument("--data-dir", required=True, help="Directory with scene_*.h5")
    parser.add_argument("--output-dir", required=True, help="Output for checkpoints & logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=32000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")
    # Use all available CPU threads for intra-op parallelism
    torch.set_num_threads(12)
    print(f"Device: {device}, PyTorch threads: {torch.get_num_threads()}", flush=True)

    # ---- Dataset ----
    print("\n=== Loading dataset ===", flush=True)
    t0 = time.time()
    full_dataset = LidarSegDataset(
        args.data_dir, SCENE_FILES,
        num_points=args.num_points, augment=True,
    )
    # Validation set WITHOUT augmentation
    val_dataset = LidarSegDataset(
        args.data_dir, SCENE_FILES,
        num_points=args.num_points, augment=False,
    )

    train_idx, val_idx = get_train_val_indices(full_dataset, VAL_SCENE)
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    print(f"Train: {len(train_subset)} frames, Val: {len(val_subset)} frames", flush=True)
    print(f"Dataset loaded in {time.time()-t0:.1f}s", flush=True)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        persistent_workers=True,
    )

    # ---- Model ----
    print("\n=== Model ===", flush=True)
    model = PointNetLiteSeg(in_channels=4, num_classes=NUM_CLASSES).to(device)
    n_params = count_parameters(model)
    print(f"PointNet-lite: {n_params:,} parameters", flush=True)

    # ---- Optimizer & Loss ----
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    print(f"\n=== Training for {args.epochs} epochs ===", flush=True)
    best_miou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t_epoch
        lr = optimizer.param_groups[0]["lr"]

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_miou": val_metrics["mean_iou"],
            "lr": lr,
            "time_s": elapsed,
        }
        for k, v in val_metrics.items():
            if k.startswith("iou_"):
                log[f"val_{k}"] = v

        history.append(log)

        # Print progress
        iou_strs = " | ".join(
            f"{CLASS_NAMES[c][:4]}={val_metrics.get(f'iou_{CLASS_NAMES[c]}', 0):.3f}"
            for c in range(1, 5)
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} mIoU={val_metrics['mean_iou']:.4f} | "
            f"{iou_strs} | "
            f"lr={lr:.6f} | {elapsed:.0f}s",
            flush=True,
        )

        # Save best model
        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": best_miou,
                "val_metrics": val_metrics,
                "n_params": n_params,
            }, ckpt_path)
            print(f"  >>> New best mIoU={best_miou:.4f}, saved to {ckpt_path}", flush=True)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, ckpt_path)

    # ---- Save history ----
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}", flush=True)

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Model: PointNet-lite ({n_params:,} params)")
    print(f"  Best val mIoU: {best_miou:.4f}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Total time: {sum(h['time_s'] for h in history):.0f}s "
          f"({sum(h['time_s'] for h in history)/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
