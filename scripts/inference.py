#!/usr/bin/env python3
"""
Inference pipeline — standalone script (no external src/ dependency).

Usage:
    # Single scene:
    python inference.py --input data/scene_8.h5 --checkpoint checkpoints_v4/best_model_v4.pt --output-dir outputs/pred_v4/

    # All HDF5 files in a directory:
    python inference.py --input data/ --checkpoint checkpoints_v4/best_model_v4.pt --output-dir outputs/pred_v4/

Output: one CSV per input file, Airbus deliverable format.
"""

import argparse
import gc
import glob
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

# ============================================================================
# CONFIG (inline — no external dependency)
# ============================================================================

NUM_CLASSES = 5
IN_CHANNELS = 5  # x, y, z, reflectivity, norm_distance

CLASS_NAMES = {0: "background", 1: "antenna", 2: "cable", 3: "electric_pole", 4: "wind_turbine"}

# Airbus spec labels for CSV output (capitalization matters)
CLASS_LABELS_CSV = {1: "Antenna", 2: "Cable", 3: "Electric Pole", 4: "Wind Turbine"}

# Internal class_id (1-4) → Airbus class_ID (0-3)
CLASS_ID_TO_AIRBUS = {1: 0, 2: 1, 3: 2, 4: 3}

DBSCAN_PARAMS = {
    1: {"eps": 2.0, "min_samples": 15},    # Antenna — inchangé
    2: {"eps": 5.0, "min_samples": 5},     # Cable — baissé (clusters petits, 13 pts médians GT)
    3: {"eps": 2.0, "min_samples": 8},     # Electric pole — baissé (21 pts médians GT)
    4: {"eps": 5.0, "min_samples": 20},    # Wind turbine — baissé légèrement
}

CABLE_MERGE_ANGLE_DEG = 15.0
CABLE_MERGE_GAP_M = 10.0

# Post-processing thresholds (calibrated from GT box stats)
# Per-class confidence threshold: min softmax probability to keep a point prediction
CONFIDENCE_THRESHOLD_PER_CLASS = {
    1: 0.40,  # antenna — bon (81 vs 111 GT)
    2: 0.27,  # cable — calibré (295 vs 285 GT)
    3: 0.25,  # electric_pole — bon (53 vs 40 GT)
    4: 0.30,  # wind_turbine — bon (56 vs 70 GT)
}
CONFIDENCE_THRESHOLD_DEFAULT = 0.3  # fallback for unknown classes

MIN_POINTS_PER_BOX = {1: 15, 2: 3, 3: 5, 4: 15}  # lowered for cable/pole/turbine (under-detected)

MAX_DIM_PER_CLASS = {  # max single dimension (m) — GT p95 + 50% margin
    1: 200.0,   # antenna (GT max=129)
    2: 400.0,   # cable (GT max=300, can be very long)
    3: 100.0,   # electric_pole (GT max=65)
    4: 250.0,   # wind_turbine (GT max=167)
}

NMS_IOU_THRESHOLD = 0.3  # suppress overlapping boxes of same class

# ============================================================================
# MODEL (inline — exact copy from training notebook v4)
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
    """PointNet v4 segmentation — multi-scale skips, ~1.88M params."""
    def __init__(self, in_channels=5, num_classes=5):
        super().__init__()
        self.enc1 = SharedMLP(in_channels, 64)
        self.enc2 = SharedMLP(64, 128)
        self.enc3 = SharedMLP(128, 256)
        self.enc4 = SharedMLP(256, 512)
        self.enc5 = SharedMLP(512, 1024)
        # 64+128+256+512+1024 = 1984
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
# HDF5 CHUNKED READER
# ============================================================================

def get_frame_boundaries(h5_path, dataset_name="lidar_points", chunk_size=2_000_000):
    """Find frame boundaries by reading in chunks — vectorized with np.diff."""
    change_indices = []
    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        n = ds.shape[0]
        prev_last_pose = None
        for offset in range(0, n, chunk_size):
            end = min(offset + chunk_size, n)
            chunk = ds[offset:end]
            ex = chunk["ego_x"]
            ey = chunk["ego_y"]
            ez = chunk["ego_z"]
            eyaw = chunk["ego_yaw"]
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
    """Read a single frame and compute features for inference.

    Returns:
        xyz_m: (N, 3) float32 — local cartesian coordinates in meters
        features: (N, 5) float32 — [x, y, z, reflectivity_norm, distance_norm]
    """
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

    features = np.concatenate([xyz, refl_norm, dist_norm], axis=1)  # (N, 5)

    del valid, dist_m, az_rad, el_rad, cos_el, x, y, z
    return xyz, features

# ============================================================================
# CHUNKED INFERENCE
# ============================================================================

@torch.no_grad()
def predict_frame(model, features_np, device, chunk_size=65536,
                  confidence_threshold=None):
    """Run inference on a full frame, chunked to avoid OOM.

    Args:
        model: PointNetSegV4 in eval mode
        features_np: (N, 5) numpy array
        device: torch device
        chunk_size: max points per forward pass
        confidence_threshold: unused (kept for API compat), per-class thresholds used instead

    Returns:
        predictions: (N,) numpy int64 — class IDs [0..4]
        confidences: (N,) numpy float32 — softmax probability of predicted class
    """
    n = len(features_np)
    predictions = np.zeros(n, dtype=np.int64)
    confidences = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = features_np[start:end]

        # Pad to at least 128 points (BatchNorm needs reasonable batch stats)
        pad_to = max(len(chunk), 128)
        if len(chunk) < pad_to:
            padded = np.zeros((pad_to, chunk.shape[1]), dtype=np.float32)
            padded[:len(chunk)] = chunk
        else:
            padded = chunk

        tensor = torch.from_numpy(padded).unsqueeze(0).to(device)  # (1, N, 5)
        logits = model(tensor)  # (1, N, 5)
        probs = F.softmax(logits[0, :len(chunk)], dim=-1)  # (N, 5)
        conf, preds = probs.max(dim=-1)
        preds = preds.cpu().numpy()
        conf = conf.cpu().numpy()

        # Per-class confidence threshold: low-confidence obstacle predictions → background
        for cid in range(1, 5):
            thresh = CONFIDENCE_THRESHOLD_PER_CLASS.get(cid, CONFIDENCE_THRESHOLD_DEFAULT)
            low_conf = (preds == cid) & (conf < thresh)
            preds[low_conf] = 0

        predictions[start:end] = preds
        confidences[start:end] = conf

        del tensor, logits, probs, preds, conf
    return predictions, confidences


@torch.no_grad()
def _get_logits_chunked(model, features_np, device, chunk_size=65536):
    """Run inference and return raw logits (N, num_classes) without argmax."""
    n = len(features_np)
    all_logits = np.zeros((n, NUM_CLASSES), dtype=np.float32)
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
        logits = model(tensor)  # (1, N, num_classes)
        all_logits[start:end] = logits[0, :len(chunk)].cpu().numpy()
        del tensor, logits
    return all_logits


def _rotate_features_z(features_np, angle_rad):
    """Rotate XYZ coordinates (first 3 columns) around Z axis. Returns copy."""
    rotated = features_np.copy()
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x, y = features_np[:, 0], features_np[:, 1]
    rotated[:, 0] = cos_a * x - sin_a * y
    rotated[:, 1] = sin_a * x + cos_a * y
    return rotated


def _rotate_logits_back(logits, angle_rad):
    """No-op for point-wise logits — rotation doesn't change logit assignment."""
    # Logits are per-point class scores, rotation doesn't change them.
    # We only needed to rotate the input features.
    return logits


@torch.no_grad()
def predict_frame_tta(model, features_np, device, chunk_size=65536,
                      confidence_threshold=None):
    """Test-Time Augmentation: average logits over 4 Z-rotations (0°, 90°, 180°, 270°).

    Returns same format as predict_frame: (predictions, confidences).
    """
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    n = len(features_np)
    avg_logits = np.zeros((n, NUM_CLASSES), dtype=np.float32)

    for angle in angles:
        if angle == 0:
            feats = features_np
        else:
            feats = _rotate_features_z(features_np, angle)
        logits = _get_logits_chunked(model, feats, device, chunk_size)
        avg_logits += logits
        del feats, logits

    avg_logits /= len(angles)

    # Softmax → argmax
    probs = np.exp(avg_logits - avg_logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    predictions = probs.argmax(axis=1).astype(np.int64)
    confidences = probs[np.arange(n), predictions].astype(np.float32)

    # Per-class confidence threshold: low-confidence obstacle predictions → background
    for cid in range(1, 5):
        thresh = CONFIDENCE_THRESHOLD_PER_CLASS.get(cid, CONFIDENCE_THRESHOLD_DEFAULT)
        low_conf = (predictions == cid) & (confidences < thresh)
        predictions[low_conf] = 0

    del avg_logits, probs
    return predictions, confidences


# ============================================================================
# CLUSTERING + BOUNDING BOXES (from build_gt_boxes.py)
# ============================================================================

def pca_oriented_bbox(points_m):
    """Compute PCA-oriented bounding box for a cluster of points."""
    center_xyz = points_m.mean(axis=0)
    centered = points_m - center_xyz
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        mins = points_m.min(axis=0)
        maxs = points_m.max(axis=0)
        return {"center_xyz": (mins + maxs) / 2.0, "dimensions": maxs - mins, "yaw": 0.0}
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        mins = points_m.min(axis=0)
        maxs = points_m.max(axis=0)
        return {"center_xyz": (mins + maxs) / 2.0, "dimensions": maxs - mins, "yaw": 0.0}
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    projected = centered @ eigenvectors
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    dimensions = maxs - mins
    box_center_pca = (mins + maxs) / 2.0
    center_xyz = center_xyz + eigenvectors @ box_center_pca
    axis1_xy = eigenvectors[:2, 0]
    yaw = np.arctan2(axis1_xy[1], axis1_xy[0])
    return {"center_xyz": center_xyz, "dimensions": dimensions, "yaw": float(yaw)}


def cluster_class_points(points_m, class_id, max_points=10000):
    """DBSCAN clustering for a single class. Returns list of point arrays."""
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
    """Merge collinear cable clusters that are close together."""
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

# ============================================================================
# POST-PROCESSING: SIZE FILTER + NMS
# ============================================================================

def filter_boxes(boxes):
    """Remove boxes that are too small (few points) or too large (over-merged)."""
    filtered = []
    for box in boxes:
        cid = box["class_id"]
        # Min points check
        if box["num_points"] < MIN_POINTS_PER_BOX.get(cid, 3):
            continue
        # Max dimension check
        max_dim = max(box["dimensions"])
        if max_dim > MAX_DIM_PER_CLASS.get(cid, 500.0):
            continue
        filtered.append(box)
    return filtered


def _box_iou_3d(box_a, box_b):
    """Approximate 3D IoU using axis-aligned overlap of PCA extents.

    Not exact for rotated boxes, but sufficient for NMS between similar objects.
    """
    ca, da = box_a["center_xyz"], box_a["dimensions"]
    cb, db = box_b["center_xyz"], box_b["dimensions"]
    # Half-extents (approximate — ignore rotation)
    ha, hb = da / 2.0, db / 2.0
    # Overlap per axis
    overlap = np.maximum(0, np.minimum(ca + ha, cb + hb) - np.maximum(ca - ha, cb - hb))
    inter = overlap[0] * overlap[1] * overlap[2]
    vol_a = da[0] * da[1] * da[2]
    vol_b = db[0] * db[1] * db[2]
    union = vol_a + vol_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_boxes(boxes, iou_threshold=NMS_IOU_THRESHOLD):
    """Non-Maximum Suppression within each class. Keep box with more points."""
    if len(boxes) <= 1:
        return boxes

    # Group by class
    by_class = {}
    for box in boxes:
        by_class.setdefault(box["class_id"], []).append(box)

    result = []
    for cid, class_boxes in by_class.items():
        # Sort by num_points descending (most confident first)
        class_boxes.sort(key=lambda b: b["num_points"], reverse=True)
        keep = []
        suppressed = [False] * len(class_boxes)
        for i in range(len(class_boxes)):
            if suppressed[i]:
                continue
            keep.append(class_boxes[i])
            for j in range(i + 1, len(class_boxes)):
                if suppressed[j]:
                    continue
                iou = _box_iou_3d(class_boxes[i], class_boxes[j])
                if iou > iou_threshold:
                    suppressed[j] = True
        result.extend(keep)
    return result


# ============================================================================
# GEOMETRIC RECLASSIFICATION
# ============================================================================

def reclassify_by_geometry(boxes):
    """Reclassify boxes based on geometric properties.

    Fixes common model confusions:
    - Antenna classified but shape is elongated + flat → likely Cable
    - Antenna classified but very large + many points → likely Wind Turbine
    """
    for box in boxes:
        if box["class_id"] != 1:  # only reclassify from antenna
            continue
        dims = box["dimensions"]
        sorted_dims = sorted(dims, reverse=True)  # [longest, middle, shortest]
        longest, middle, shortest = sorted_dims

        # Elongated + flat → Cable (ratio length/width > 5, height < 1m)
        if middle > 0 and longest / middle > 5.0 and shortest < 1.0:
            box["class_id"] = 2
            box["class_label"] = CLASS_LABELS_CSV[2]

        # Very large + many points → Wind Turbine (any dim > 15m and > 200 pts)
        elif longest > 15.0 and box["num_points"] > 200:
            box["class_id"] = 4
            box["class_label"] = CLASS_LABELS_CSV[4]

    return boxes

# ============================================================================
# PREDICTIONS → BOUNDING BOXES
# ============================================================================

def predictions_to_boxes(xyz_m, predictions, confidences=None,
                         use_per_class_conf=True):
    """Convert per-point predictions to bounding boxes via DBSCAN + PCA + post-processing.

    Pipeline: cluster → PCA bbox → geometric reclassification → per-class confidence filter → size filter → NMS

    Args:
        xyz_m: (N, 3) point positions in meters
        predictions: (N,) class IDs [0..4]
        confidences: (N,) softmax probability of predicted class (optional)
        use_per_class_conf: if True, use BOX_CONFIDENCE_THRESHOLD_PER_CLASS (default)

    Returns:
        list of dicts with keys: center_xyz, dimensions, yaw, class_id, class_label, num_points, confidence
    """
    boxes = []
    for cid in range(1, 5):
        mask = predictions == cid
        n_pts = mask.sum()
        if n_pts == 0:
            continue

        class_points = xyz_m[mask]
        class_conf = confidences[mask] if confidences is not None else None

        clusters = cluster_class_points(class_points, cid)

        if cid == 2 and len(clusters) > 1:
            clusters = merge_cable_clusters(clusters)

        # Build BallTree once per class for confidence lookup
        conf_tree = None
        if class_conf is not None and len(clusters) > 0:
            from sklearn.neighbors import BallTree
            conf_tree = BallTree(class_points)

        for pts in clusters:
            if len(pts) < 3:
                continue

            # Compute mean confidence for this cluster
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

    # Geometric reclassification (before confidence filter — fixes antenna→cable/turbine)
    boxes = reclassify_by_geometry(boxes)

    # Per-class box confidence filter
    if use_per_class_conf:
        filtered = []
        for b in boxes:
            thresh = BOX_CONFIDENCE_THRESHOLD_PER_CLASS.get(
                b["class_id"], BOX_CONFIDENCE_THRESHOLD_DEFAULT)
            if b["confidence"] >= thresh:
                filtered.append(b)
        boxes = filtered

    # Post-processing
    boxes = filter_boxes(boxes)
    boxes = nms_boxes(boxes)
    return boxes

# ============================================================================
# CSV OUTPUT
# ============================================================================

# Airbus deliverable format
CSV_HEADER = ("ego_x,ego_y,ego_z,ego_yaw,"
              "bbox_center_x,bbox_center_y,bbox_center_z,"
              "bbox_width,bbox_length,bbox_height,"
              "bbox_yaw,"
              "class_ID,class_label\n")


def boxes_to_csv_lines(boxes, ego_x, ego_y, ego_z, ego_yaw):
    """Format boxes as CSV lines (Airbus deliverable format)."""
    lines = []
    for box in boxes:
        c = box["center_xyz"]
        d = box["dimensions"]
        airbus_cid = CLASS_ID_TO_AIRBUS[box["class_id"]]
        lines.append(
            f"{ego_x},{ego_y},{ego_z},{ego_yaw},"
            f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
            f"{d[0]:.4f},{d[1]:.4f},{d[2]:.4f},"
            f"{box['yaw']:.4f},"
            f"{airbus_cid},{box['class_label']}\n"
        )
    return lines

# ============================================================================
# MAIN PIPELINE
# ============================================================================

# Per-class box confidence threshold (calibrated on scene_8 val)
BOX_CONFIDENCE_THRESHOLD_PER_CLASS = {
    1: 0.70,  # antenna — bon (81 boxes)
    2: 0.55,  # cable — calibré (295 vs 285 GT)
    3: 0.45,  # electric_pole — bon (53 boxes)
    4: 0.60,  # wind_turbine — bon (56 boxes)
}
BOX_CONFIDENCE_THRESHOLD_DEFAULT = 0.6  # fallback

def run_inference(input_path, checkpoint_path, output_dir, chunk_size, device_str,
                  use_tta=False):
    """Run full inference pipeline on one or multiple HDF5 files."""

    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from {checkpoint_path}...", flush=True)
    model = PointNetSegV4(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointNetSegV4: {n_params:,} params on {device}", flush=True)
    if "val_obstacle_miou" in ckpt:
        print(f"Checkpoint epoch {ckpt.get('epoch', '?')}, val obstacle mIoU={ckpt['val_obstacle_miou']:.4f}", flush=True)

    # Print per-class thresholds
    print(f"\nPer-class conf thresholds (point): {CONFIDENCE_THRESHOLD_PER_CLASS}", flush=True)
    print(f"Per-class conf thresholds (box):   {BOX_CONFIDENCE_THRESHOLD_PER_CLASS}", flush=True)

    # --- Collect input files ---
    if os.path.isdir(input_path):
        h5_files = sorted(glob.glob(os.path.join(input_path, "*.h5")))
    else:
        h5_files = [input_path]

    if not h5_files:
        print(f"ERROR: No .h5 files found in {input_path}")
        sys.exit(1)

    predict_fn = predict_frame_tta if use_tta else predict_frame
    print(f"Input: {len(h5_files)} file(s), TTA={'ON' if use_tta else 'OFF'}", flush=True)

    total_boxes = 0
    total_frames = 0
    t_total = time.time()

    for h5_path in h5_files:
        scene_name = os.path.splitext(os.path.basename(h5_path))[0]
        output_csv = os.path.join(output_dir, f"{scene_name}.csv")

        print(f"\n{'='*60}", flush=True)
        print(f"[{scene_name}] Processing {h5_path}", flush=True)

        # Find frame boundaries
        t0 = time.time()
        frames_info = get_frame_boundaries(h5_path)
        n_frames = len(frames_info)
        print(f"[{scene_name}] {n_frames} frames found ({time.time()-t0:.1f}s)", flush=True)

        # Write CSV header
        with open(output_csv, "w") as f:
            f.write(CSV_HEADER)

        scene_boxes = 0
        class_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for idx in range(n_frames):
            start, end, ego_x, ego_y, ego_z, ego_yaw = frames_info[idx]

            # Read frame
            xyz_m, features = read_frame_for_inference(h5_path, start, end)
            if len(xyz_m) == 0:
                total_frames += 1
                continue

            # Predict (per-class thresholds applied internally)
            predictions, confidences = predict_fn(
                model, features, device, chunk_size=chunk_size)
            del features

            # Predictions → boxes (per-class box confidence + geometric reclassification)
            boxes = predictions_to_boxes(xyz_m, predictions, confidences)
            del xyz_m, predictions, confidences

            # Write CSV
            if boxes:
                lines = boxes_to_csv_lines(boxes, ego_x, ego_y, ego_z, ego_yaw)
                with open(output_csv, "a") as f:
                    f.writelines(lines)
                scene_boxes += len(boxes)
                for box in boxes:
                    class_counts[box["class_id"]] += 1

            del boxes
            gc.collect()
            total_frames += 1

            if (idx + 1) % 10 == 0 or idx == n_frames - 1:
                print(f"  [{scene_name}] {idx+1}/{n_frames} frames, {scene_boxes} boxes", flush=True)

        total_boxes += scene_boxes
        print(f"[{scene_name}] DONE — {scene_boxes} boxes from {n_frames} frames", flush=True)
        print(f"  Per class: " + ", ".join(
            f"{CLASS_NAMES[c]}={class_counts[c]}" for c in range(1, 5)
        ), flush=True)
        print(f"  Output: {output_csv}", flush=True)

        del frames_info
        gc.collect()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"INFERENCE COMPLETE", flush=True)
    print(f"Total: {total_boxes} boxes across {total_frames} frames", flush=True)
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Avg: {elapsed/max(total_frames,1):.2f}s/frame, {total_boxes/max(total_frames,1):.1f} boxes/frame", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="LiDAR obstacle detection inference — PointNetSegV4 (v7 per-class thresholds)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scene:
  python inference.py --input data/scene_8.h5 --checkpoint best_model_v5.pt --output-dir outputs/pred/

  # All scenes in a directory:
  python inference.py --input data/ --checkpoint best_model_v5.pt --output-dir outputs/pred/

  # With TTA (4x slower but more robust):
  python inference.py --input data/ --checkpoint best_model_v5.pt --output-dir outputs/pred/ --tta

  # CPU inference:
  python inference.py --input data/scene_8.h5 --checkpoint best_model_v5.pt --output-dir outputs/pred/ --device cpu

Per-class thresholds are hardcoded in the CONFIG section at the top of this file.
""")
    parser.add_argument("--input", required=True,
                        help="Path to a single .h5 file or directory containing .h5 files")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for CSV predictions")
    parser.add_argument("--chunk-size", type=int, default=65536,
                        help="Max points per forward pass (default: 65536)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation (4x Z-rotations, slower but more robust)")
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cuda', 'cpu' (default: auto)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference(args.input, args.checkpoint, args.output_dir,
                  args.chunk_size, device, args.tta)


if __name__ == "__main__":
    main()
