"""
PointNet-lite: lightweight per-point segmentation network.
~200k parameters. Input: (B, N, 4) → Output: (B, N, 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """1D conv acting as shared MLP across points."""

    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=not bn)
        self.bn = nn.BatchNorm1d(out_ch) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class PointNetLiteSeg(nn.Module):
    """Lightweight PointNet for per-point segmentation.

    Architecture:
        Input (B, N, 4) → transpose → (B, 4, N)
        Local features:  4 → 64 → 128 → 256
        Global feature:  max pool → 256
        Concat:          256 + 256 = 512 per point
        Segmentation:    512 → 128 → 64 → num_classes

    ~200k params for num_classes=5
    """

    def __init__(self, in_channels=4, num_classes=5):
        super().__init__()

        # Local feature extraction
        self.local1 = SharedMLP(in_channels, 64)
        self.local2 = SharedMLP(64, 128)
        self.local3 = SharedMLP(128, 256)

        # Segmentation head (per-point)
        self.seg1 = SharedMLP(256 + 256, 128)
        self.seg2 = SharedMLP(128, 64)
        self.seg3 = nn.Conv1d(64, num_classes, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: (B, N, in_channels) point features
        Returns:
            logits: (B, N, num_classes) per-point class logits
        """
        B, N, _ = x.shape
        x = x.transpose(1, 2)  # (B, C, N)

        # Local features
        local1 = self.local1(x)     # (B, 64, N)
        local2 = self.local2(local1) # (B, 128, N)
        local3 = self.local3(local2) # (B, 256, N)

        # Global feature via max pooling
        global_feat = local3.max(dim=2, keepdim=True)[0]  # (B, 256, 1)
        global_feat = global_feat.expand(-1, -1, N)        # (B, 256, N)

        # Concatenate local + global
        concat = torch.cat([local3, global_feat], dim=1)   # (B, 512, N)

        # Segmentation head
        seg = self.seg1(concat)      # (B, 128, N)
        seg = self.dropout(seg)
        seg = self.seg2(seg)         # (B, 64, N)
        seg = self.seg3(seg)         # (B, num_classes, N)

        return seg.transpose(1, 2)   # (B, N, num_classes)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PointNetLiteSeg(in_channels=4, num_classes=5)
    print(f"PointNet-lite Segmentation")
    print(f"  Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 32000, 4)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Output classes: {out.argmax(dim=-1).unique().tolist()}")

    # Layer-by-layer param count
    print("\n  Layer breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name:40s} {param.numel():>8,} params  {list(param.shape)}")
