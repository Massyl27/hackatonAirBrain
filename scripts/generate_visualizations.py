#!/usr/bin/env python3
"""
Generate visualization screenshots — Livrable #4 Airbus.

Produces up to 10 PNG images showing point clouds with predicted 3D bounding boxes
colored by class. Two views per frame: top-down (XY) and side view (XZ).

Usage:
    python scripts/generate_visualizations.py \
        --input data/scene_8.h5 \
        --checkpoint checkpoints_v5/best_model_v5.pt \
        --output-dir outputs/visualizations/ \
        --num-frames 10
"""

import argparse
import gc
import os
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

# ============================================================================
# CONFIG
# ============================================================================

NUM_CLASSES = 5
IN_CHANNELS = 5

CLASS_NAMES = {0: "Background", 1: "Antenna", 2: "Cable", 3: "Electric pole", 4: "Wind turbine"}

CLASS_LABELS_CSV = {1: "Antenna", 2: "Cable", 3: "Electric Pole", 4: "Wind Turbine"}

# Internal class_id (1-4) → Airbus class_ID (0-3)
CLASS_ID_TO_AIRBUS = {1: 0, 2: 1, 3: 2, 4: 3}

# Colors for visualization (RGBA)
CLASS_COLORS = {
    0: (0.7, 0.7, 0.7, 0.05),    # background — very transparent grey
    1: (0.15, 0.09, 0.71, 0.9),   # antenna — blue (from RGB 38,23,180)
    2: (0.69, 0.52, 0.18, 0.9),   # cable — gold (from RGB 177,132,47)
    3: (0.51, 0.32, 0.38, 0.9),   # electric pole — mauve (from RGB 129,81,97)
    4: (0.26, 0.52, 0.04, 0.9),   # wind turbine — green (from RGB 66,132,9)
}

# Bounding box edge colors (opaque, vivid)
BOX_COLORS = {
    1: "#2617B4",   # antenna — blue
    2: "#B18430",   # cable — gold
    3: "#815161",   # electric pole — mauve
    4: "#428409",   # wind turbine — green
}

DBSCAN_PARAMS = {
    1: {"eps": 2.0, "min_samples": 15},    # Antenna — inchangé
    2: {"eps": 5.0, "min_samples": 5},     # Cable — baissé (clusters petits)
    3: {"eps": 2.0, "min_samples": 8},     # Electric pole — baissé
    4: {"eps": 5.0, "min_samples": 20},    # Wind turbine — baissé légèrement
}

CABLE_MERGE_ANGLE_DEG = 15.0
CABLE_MERGE_GAP_M = 10.0

# Per-class confidence thresholds
CONFIDENCE_THRESHOLD_PER_CLASS = {
    1: 0.40,  # antenna — bon (81 vs 111 GT)
    2: 0.27,  # cable — calibré (295 vs 285 GT)
    3: 0.25,  # electric_pole — bon (53 vs 40 GT)
    4: 0.30,  # wind_turbine — bon (56 vs 70 GT)
}
CONFIDENCE_THRESHOLD_DEFAULT = 0.3

BOX_CONFIDENCE_THRESHOLD_PER_CLASS = {
    1: 0.70,  # antenna — bon (81 boxes)
    2: 0.55,  # cable — calibré (295 vs 285 GT)
    3: 0.45,  # electric_pole — bon (53 boxes)
    4: 0.60,  # wind_turbine — bon (56 boxes)
}
BOX_CONFIDENCE_THRESHOLD_DEFAULT = 0.6

MIN_POINTS_PER_BOX = {1: 15, 2: 3, 3: 5, 4: 15}
MAX_DIM_PER_CLASS = {1: 200.0, 2: 400.0, 3: 100.0, 4: 250.0}
NMS_IOU_THRESHOLD = 0.3

# ============================================================================
# MODEL (inline copy)
# ============================================================================

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

# ============================================================================
# HDF5 READER
# ============================================================================

def get_frame_boundaries(h5_path, dataset_name="lidar_points", chunk_size=2_000_000):
    change_indices = []
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        n = ds.shape[0]
        prev_last_pose = None
        for offset in range(0, n, chunk_size):
            end = min(offset + chunk_size, n)
            chunk = ds[offset:end]
            ex, ey, ez, eyaw = chunk["ego_x"], chunk["ego_y"], chunk["ego_z"], chunk["ego_yaw"]
            if prev_last_pose is not None:
                cur_first = (int(ex[0]), int(ey[0]), int(ez[0]), int(eyaw[0]))
                if cur_first != prev_last_pose:
                    change_indices.append(offset)
            changes = np.where(
                (np.diff(ex) != 0) | (np.diff(ey) != 0) |
                (np.diff(ez) != 0) | (np.diff(eyaw) != 0)
            )[0] + 1
            for c in changes:
                change_indices.append(offset + int(c))
            prev_last_pose = (int(ex[-1]), int(ey[-1]), int(ez[-1]), int(eyaw[-1]))
            del chunk, ex, ey, ez, eyaw
            gc.collect()
    starts = [0] + change_indices
    ends = change_indices + [n]
    frames = []
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        for s, e in zip(starts, ends):
            row = ds[s]
            frames.append((s, e, int(row["ego_x"]), int(row["ego_y"]),
                           int(row["ego_z"]), int(row["ego_yaw"])))
    return frames


def read_frame_for_inference(h5_path, start, end, dataset_name="lidar_points"):
    with h5py.File(h5_path, "r") as f:
        chunk = f[dataset_name][start:end]
    valid = chunk[chunk["distance_cm"] > 0]
    del chunk
    dist_m = valid["distance_cm"].astype(np.float64) / 100.0
    az_rad = np.radians(valid["azimuth_raw"].astype(np.float64) / 100.0)
    el_rad = np.radians(valid["elevation_raw"].astype(np.float64) / 100.0)
    cos_el = np.cos(el_rad)
    x = dist_m * cos_el * np.cos(az_rad)
    y = -dist_m * cos_el * np.sin(az_rad)
    z = dist_m * np.sin(el_rad)
    xyz = np.column_stack((x, y, z)).astype(np.float32)
    refl_norm = (valid["reflectivity"].astype(np.float32) / 255.0).reshape(-1, 1)
    dist_norm = (dist_m.astype(np.float32) / 300.0).reshape(-1, 1)
    features = np.concatenate([xyz, refl_norm, dist_norm], axis=1)

    # Also read GT labels for comparison
    r, g, b = valid["r"].astype(int), valid["g"].astype(int), valid["b"].astype(int)
    gt_labels = np.zeros(len(valid), dtype=np.int64)
    gt_labels[(r == 38) & (g == 23) & (b == 180)] = 1   # antenna
    gt_labels[(r == 177) & (g == 132) & (b == 47)] = 2   # cable
    gt_labels[(r == 129) & (g == 81) & (b == 97)] = 3    # electric_pole
    gt_labels[(r == 66) & (g == 132) & (b == 9)] = 4     # wind_turbine

    del valid, dist_m, az_rad, el_rad, cos_el, x, y, z
    return xyz, features, gt_labels

# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def predict_frame(model, features_np, device, chunk_size=65536):
    n = len(features_np)
    predictions = np.zeros(n, dtype=np.int64)
    confidences = np.zeros(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = features_np[start:end]
        pad_to = max(len(chunk), 128)
        if len(chunk) < pad_to:
            padded = np.zeros((pad_to, chunk.shape[1]), dtype=np.float32)
            padded[:len(chunk)] = chunk
        else:
            padded = chunk
        tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits[0, :len(chunk)], dim=-1)
        conf, preds = probs.max(dim=-1)
        preds = preds.cpu().numpy()
        conf = conf.cpu().numpy()
        # Per-class confidence threshold
        for cid in range(1, 5):
            thresh = CONFIDENCE_THRESHOLD_PER_CLASS.get(cid, CONFIDENCE_THRESHOLD_DEFAULT)
            low_conf = (preds == cid) & (conf < thresh)
            preds[low_conf] = 0
        predictions[start:end] = preds
        confidences[start:end] = conf
        del tensor, logits, probs, preds, conf
    return predictions, confidences

# ============================================================================
# CLUSTERING + BOUNDING BOXES
# ============================================================================

def pca_oriented_bbox(points_m):
    center_xyz = points_m.mean(axis=0)
    centered = points_m - center_xyz
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        mins, maxs = points_m.min(axis=0), points_m.max(axis=0)
        return {"center_xyz": (mins + maxs) / 2.0, "dimensions": maxs - mins, "yaw": 0.0}
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        mins, maxs = points_m.min(axis=0), points_m.max(axis=0)
        return {"center_xyz": (mins + maxs) / 2.0, "dimensions": maxs - mins, "yaw": 0.0}
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    projected = centered @ eigenvectors
    mins, maxs = projected.min(axis=0), projected.max(axis=0)
    dimensions = maxs - mins
    box_center_pca = (mins + maxs) / 2.0
    center_xyz = center_xyz + eigenvectors @ box_center_pca
    axis1_xy = eigenvectors[:2, 0]
    yaw = np.arctan2(axis1_xy[1], axis1_xy[0])
    return {"center_xyz": center_xyz, "dimensions": dimensions, "yaw": float(yaw)}


def cluster_class_points(points_m, class_id, max_points=10000):
    params = DBSCAN_PARAMS[class_id]
    eps, min_samples = params["eps"], params["min_samples"]
    if len(points_m) < min_samples:
        return []
    full_points = points_m
    if len(points_m) > max_points:
        idx = np.random.choice(len(points_m), max_points, replace=False)
        points_m = points_m[idx]
    labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree").fit_predict(points_m)
    if len(full_points) > max_points:
        from sklearn.neighbors import BallTree
        sampled_mask = labels >= 0
        if sampled_mask.sum() == 0:
            return []
        tree = BallTree(points_m[sampled_mask])
        _, indices = tree.query(full_points, k=1)
        full_labels = labels[sampled_mask][indices.ravel()]
        dists = np.linalg.norm(full_points - points_m[sampled_mask][indices.ravel()], axis=1)
        full_labels[dists > eps * 2] = -1
        labels = full_labels
        points_m = full_points
    clusters = []
    for lbl in sorted(set(labels) - {-1}):
        clusters.append(points_m[labels == lbl])
    return clusters


def merge_cable_clusters(clusters):
    if len(clusters) <= 1:
        return clusters
    angle_thresh = np.radians(CABLE_MERGE_ANGLE_DEG)
    gap_thresh = CABLE_MERGE_GAP_M
    infos = []
    for pts in clusters:
        if len(pts) < 4:
            infos.append({"points": pts, "center": pts.mean(axis=0), "axis1": None})
            continue
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            infos.append({"points": pts, "center": pts.mean(axis=0), "axis1": None})
            continue
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            infos.append({"points": pts, "center": pts.mean(axis=0), "axis1": None})
            continue
        axis1 = eigvecs[:, eigvals.argsort()[::-1][0]]
        if axis1[0] < 0:
            axis1 = -axis1
        infos.append({"points": pts, "center": pts.mean(axis=0), "axis1": axis1})
    merged_flags = [False] * len(infos)
    result = []
    for i in range(len(infos)):
        if merged_flags[i]:
            continue
        current = infos[i]["points"]
        if infos[i]["axis1"] is not None:
            for j in range(i + 1, len(infos)):
                if merged_flags[j] or infos[j]["axis1"] is None:
                    continue
                dot = min(abs(np.dot(infos[i]["axis1"], infos[j]["axis1"])), 1.0)
                if np.arccos(dot) > angle_thresh:
                    continue
                cdist = np.linalg.norm(infos[i]["center"] - infos[j]["center"])
                ext_i = np.abs((infos[i]["points"] - infos[i]["center"]) @ infos[i]["axis1"]).max()
                ext_j = np.abs((infos[j]["points"] - infos[j]["center"]) @ infos[j]["axis1"]).max()
                if cdist - ext_i - ext_j <= gap_thresh:
                    current = np.vstack([current, infos[j]["points"]])
                    merged_flags[j] = True
        result.append(current)
    return result


def filter_boxes(boxes):
    filtered = []
    for box in boxes:
        cid = box["class_id"]
        if box["num_points"] < MIN_POINTS_PER_BOX.get(cid, 3):
            continue
        if max(box["dimensions"]) > MAX_DIM_PER_CLASS.get(cid, 500.0):
            continue
        filtered.append(box)
    return filtered


def _box_iou_3d(box_a, box_b):
    ca, da = box_a["center_xyz"], box_a["dimensions"]
    cb, db = box_b["center_xyz"], box_b["dimensions"]
    ha, hb = da / 2.0, db / 2.0
    overlap = np.maximum(0, np.minimum(ca + ha, cb + hb) - np.maximum(ca - ha, cb - hb))
    inter = overlap[0] * overlap[1] * overlap[2]
    vol_a, vol_b = da[0] * da[1] * da[2], db[0] * db[1] * db[2]
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, iou_threshold=NMS_IOU_THRESHOLD):
    if len(boxes) <= 1:
        return boxes
    by_class = {}
    for box in boxes:
        by_class.setdefault(box["class_id"], []).append(box)
    result = []
    for cid, class_boxes in by_class.items():
        class_boxes.sort(key=lambda b: b["num_points"], reverse=True)
        suppressed = [False] * len(class_boxes)
        for i in range(len(class_boxes)):
            if suppressed[i]:
                continue
            result.append(class_boxes[i])
            for j in range(i + 1, len(class_boxes)):
                if not suppressed[j] and _box_iou_3d(class_boxes[i], class_boxes[j]) > iou_threshold:
                    suppressed[j] = True
    return result


def reclassify_by_geometry(boxes):
    """Reclassify boxes based on geometric properties."""
    for box in boxes:
        if box["class_id"] != 1:
            continue
        dims = box["dimensions"]
        sorted_dims = sorted(dims, reverse=True)
        longest, middle, shortest = sorted_dims
        if middle > 0 and longest / middle > 5.0 and shortest < 1.0:
            box["class_id"] = 2
            box["class_label"] = CLASS_LABELS_CSV[2]
        elif longest > 15.0 and box["num_points"] > 200:
            box["class_id"] = 4
            box["class_label"] = CLASS_LABELS_CSV[4]
    return boxes


def predictions_to_boxes(xyz_m, predictions, confidences=None):
    boxes = []
    for cid in range(1, 5):
        mask = predictions == cid
        if mask.sum() == 0:
            continue
        class_points = xyz_m[mask]
        class_conf = confidences[mask] if confidences is not None else None
        clusters = cluster_class_points(class_points, cid)
        if cid == 2 and len(clusters) > 1:
            clusters = merge_cable_clusters(clusters)
        conf_tree = None
        if class_conf is not None and len(clusters) > 0:
            from sklearn.neighbors import BallTree
            conf_tree = BallTree(class_points)
        for pts in clusters:
            if len(pts) < 3:
                continue
            box_confidence = 0.0
            if conf_tree is not None:
                _, indices = conf_tree.query(pts, k=1)
                box_confidence = float(class_conf[indices.ravel()].mean())
            bbox = pca_oriented_bbox(pts)
            boxes.append({
                "center_xyz": bbox["center_xyz"],
                "dimensions": bbox["dimensions"],
                "yaw": bbox["yaw"],
                "class_id": cid,
                "class_label": CLASS_LABELS_CSV[cid],
                "num_points": len(pts),
                "confidence": box_confidence,
            })
    # Geometric reclassification
    boxes = reclassify_by_geometry(boxes)
    # Per-class box confidence filter
    filtered = []
    for b in boxes:
        thresh = BOX_CONFIDENCE_THRESHOLD_PER_CLASS.get(
            b["class_id"], BOX_CONFIDENCE_THRESHOLD_DEFAULT)
        if b["confidence"] >= thresh:
            filtered.append(b)
    boxes = filtered
    boxes = filter_boxes(boxes)
    boxes = nms_boxes(boxes)
    return boxes

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_rotated_box_2d(ax, cx, cy, w, h, yaw, color, linewidth=2):
    """Draw a rotated rectangle on a 2D matplotlib axes."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    # 4 corners of the box before rotation
    corners = np.array([
        [-w/2, -h/2],
        [+w/2, -h/2],
        [+w/2, +h/2],
        [-w/2, +h/2],
        [-w/2, -h/2],  # close the box
    ])
    # Rotate
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    rotated = corners @ rot.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=linewidth, solid_capstyle='round')


def render_frame(xyz_m, predictions, boxes, frame_idx, ego_info, output_path,
                 max_display_points=100000):
    """Render a frame with two views: top-down (XY) and side (XZ)."""

    # Subsample points for display
    if len(xyz_m) > max_display_points:
        idx = np.random.choice(len(xyz_m), max_display_points, replace=False)
        xyz_disp = xyz_m[idx]
        pred_disp = predictions[idx]
    else:
        xyz_disp = xyz_m
        pred_disp = predictions

    # Assign colors to points
    colors = np.array([CLASS_COLORS[p] for p in pred_disp])

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    ego_x, ego_y, ego_z, ego_yaw = ego_info

    # --- Top-down view (XY) ---
    ax = axes[0]
    # Background points first (below), then obstacle points on top
    bg_mask = pred_disp == 0
    obs_mask = ~bg_mask

    ax.scatter(xyz_disp[bg_mask, 0], xyz_disp[bg_mask, 1],
               c=colors[bg_mask], s=0.1, rasterized=True)
    ax.scatter(xyz_disp[obs_mask, 0], xyz_disp[obs_mask, 1],
               c=colors[obs_mask], s=1.5, rasterized=True)

    # Draw bounding boxes (top-down: use X, Y, width, length, yaw)
    for box in boxes:
        c = box["center_xyz"]
        d = box["dimensions"]
        draw_rotated_box_2d(ax, c[0], c[1], d[0], d[1], box["yaw"],
                           color=BOX_COLORS[box["class_id"]], linewidth=2.5)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Top-down view (XY)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- Side view (XZ) ---
    ax = axes[1]
    ax.scatter(xyz_disp[bg_mask, 0], xyz_disp[bg_mask, 2],
               c=colors[bg_mask], s=0.1, rasterized=True)
    ax.scatter(xyz_disp[obs_mask, 0], xyz_disp[obs_mask, 2],
               c=colors[obs_mask], s=1.5, rasterized=True)

    # Draw bounding boxes (side: use X, Z, width, height, no rotation)
    for box in boxes:
        c = box["center_xyz"]
        d = box["dimensions"]
        # Side view: axis-aligned rectangle (X, Z)
        rect = patches.Rectangle(
            (c[0] - d[0]/2, c[2] - d[2]/2), d[0], d[2],
            linewidth=2.5, edgecolor=BOX_COLORS[box["class_id"]],
            facecolor="none", linestyle="-"
        )
        ax.add_patch(rect)

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Z (m)", fontsize=11)
    ax.set_title("Side view (XZ)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=BOX_COLORS[1], linewidth=3, label=f"Antenna ({sum(1 for b in boxes if b['class_id']==1)})"),
        Line2D([0], [0], color=BOX_COLORS[2], linewidth=3, label=f"Cable ({sum(1 for b in boxes if b['class_id']==2)})"),
        Line2D([0], [0], color=BOX_COLORS[3], linewidth=3, label=f"Electric pole ({sum(1 for b in boxes if b['class_id']==3)})"),
        Line2D([0], [0], color=BOX_COLORS[4], linewidth=3, label=f"Wind turbine ({sum(1 for b in boxes if b['class_id']==4)})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=11,
               frameon=True, fancybox=True, shadow=True)

    # Title
    n_obs = sum(1 for p in predictions if p > 0)
    fig.suptitle(
        f"Frame {frame_idx} — {len(boxes)} detections — "
        f"{n_obs:,} obstacle pts / {len(predictions):,} total — "
        f"PointNetSegV4 (v5, mIoU=0.212)",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}", flush=True)


# ============================================================================
# FRAME SELECTION — pick diverse frames
# ============================================================================

def select_diverse_frames(all_frame_results, num_frames=10):
    """Select frames that showcase different class combinations and box counts."""
    if len(all_frame_results) <= num_frames:
        return all_frame_results

    # Sort by number of distinct classes detected (descending), then by box count
    scored = []
    for fr in all_frame_results:
        classes_present = set(b["class_id"] for b in fr["boxes"])
        scored.append((len(classes_present), len(fr["boxes"]), fr))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = []
    seen_class_combos = set()

    # First pass: pick frames with unique class combinations
    for n_cls, n_box, fr in scored:
        if len(selected) >= num_frames:
            break
        combo = frozenset(b["class_id"] for b in fr["boxes"])
        if combo not in seen_class_combos:
            selected.append(fr)
            seen_class_combos.add(combo)

    # Second pass: fill remaining slots with highest box count frames
    if len(selected) < num_frames:
        selected_indices = set(fr["frame_idx"] for fr in selected)
        for n_cls, n_box, fr in scored:
            if len(selected) >= num_frames:
                break
            if fr["frame_idx"] not in selected_indices:
                selected.append(fr)
                selected_indices.add(fr["frame_idx"])

    return selected


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate visualization screenshots (Livrable #4)")
    parser.add_argument("--input", required=True, help="Path to .h5 file")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--output-dir", default="outputs/visualizations/", help="Output directory for PNGs")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to visualize")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...", flush=True)
    model = PointNetSegV4(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointNetSegV4: {n_params:,} params on {device}", flush=True)

    # Get frame boundaries
    print(f"\nReading frame boundaries from {args.input}...", flush=True)
    frames_info = get_frame_boundaries(args.input)
    n_frames = len(frames_info)
    print(f"Found {n_frames} frames", flush=True)

    # Run inference on ALL frames to find the best ones
    print(f"\nRunning inference on all {n_frames} frames to select best {args.num_frames}...", flush=True)
    all_results = []
    t0 = time.time()

    for idx in range(n_frames):
        start, end, ego_x, ego_y, ego_z, ego_yaw = frames_info[idx]
        xyz_m, features, gt_labels = read_frame_for_inference(args.input, start, end)
        if len(xyz_m) == 0:
            continue

        predictions, confidences = predict_frame(model, features, device)
        boxes = predictions_to_boxes(xyz_m, predictions, confidences)

        if boxes:  # Only consider frames with detections
            all_results.append({
                "frame_idx": idx,
                "xyz_m": xyz_m,
                "predictions": predictions,
                "boxes": boxes,
                "ego_info": (ego_x, ego_y, ego_z, ego_yaw),
            })
        else:
            del xyz_m, predictions
            gc.collect()

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{n_frames} frames processed, {len(all_results)} with detections", flush=True)

    elapsed = time.time() - t0
    print(f"\nInference done: {len(all_results)} frames with detections ({elapsed:.0f}s)", flush=True)

    # Select diverse frames
    selected = select_diverse_frames(all_results, args.num_frames)
    print(f"Selected {len(selected)} frames for visualization", flush=True)

    # Free unselected frames
    selected_indices = set(fr["frame_idx"] for fr in selected)
    for fr in all_results:
        if fr["frame_idx"] not in selected_indices:
            del fr["xyz_m"], fr["predictions"], fr["boxes"]
    gc.collect()

    # Render selected frames
    print(f"\nRendering visualizations...", flush=True)
    scene_name = os.path.splitext(os.path.basename(args.input))[0]

    for i, fr in enumerate(selected):
        filename = f"{scene_name}_frame{fr['frame_idx']:03d}.png"
        output_path = os.path.join(args.output_dir, filename)

        classes_in_frame = sorted(set(b["class_id"] for b in fr["boxes"]))
        class_names = [CLASS_NAMES[c] for c in classes_in_frame]
        print(f"\n  [{i+1}/{len(selected)}] Frame {fr['frame_idx']}: "
              f"{len(fr['boxes'])} boxes ({', '.join(class_names)})", flush=True)

        render_frame(
            fr["xyz_m"], fr["predictions"], fr["boxes"],
            fr["frame_idx"], fr["ego_info"], output_path
        )

        del fr["xyz_m"], fr["predictions"]
        gc.collect()

    print(f"\nDone! {len(selected)} visualizations saved to {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
