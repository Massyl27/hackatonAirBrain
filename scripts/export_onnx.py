#!/usr/bin/env python3
"""Export PointNetSegV4 to ONNX format."""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model (inline copy)
class SharedMLP(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=not bn)
        self.bn = nn.BatchNorm1d(out_ch) if bn else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class PointNetSegV4(nn.Module):
    def __init__(self, in_channels=5, num_classes=5):
        super().__init__()
        self.enc1 = SharedMLP(in_channels, 64)
        self.enc2 = SharedMLP(64, 128)
        self.enc3 = SharedMLP(128, 256)
        self.enc4 = SharedMLP(256, 512)
        self.enc5 = SharedMLP(512, 1024)
        self.seg1 = SharedMLP(64 + 128 + 256 + 512 + 1024, 512)
        self.seg2 = SharedMLP(512, 256)
        self.seg3 = SharedMLP(256, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.head = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(1, 2)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        g = e5.max(dim=2, keepdim=True)[0].expand(-1, -1, N)
        seg = torch.cat([e1, e2, e3, e4, g], dim=1)
        seg = self.seg1(seg)
        seg = self.dropout1(seg)
        seg = self.seg2(seg)
        seg = self.dropout2(seg)
        seg = self.seg3(seg)
        seg = self.head(seg)
        return seg.transpose(1, 2)


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_v5/best_model_v5.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_v5/pointnet_seg_v5.onnx"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model = PointNetSegV4(in_channels=5, num_classes=5)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    if "val_obstacle_miou" in ckpt:
        print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, obstacle mIoU={ckpt['val_obstacle_miou']:.4f}")

    dummy = torch.randn(1, 65536, 5)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["points"],
        output_names=["logits"],
        dynamic_axes={"points": {1: "num_points"}, "logits": {1: "num_points"}},
        opset_version=17,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX exported: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
