#!/usr/bin/env python3
"""
Build GT bounding boxes for all frames — standalone script (no Colab/GPU needed).

Usage:
    python build_gt_boxes.py --data-dir /path/to/data --output-dir /path/to/output

Data dir should contain scene_1.h5 ... scene_10.h5
Output: gt_boxes_all.csv + frame_box_counts.csv
"""

import argparse
import gc
import os
import sys
import time

import h5py
import numpy as np
from sklearn.cluster import DBSCAN

# ============================================================================
# CONFIG (inline — no external dependency)
# ============================================================================

CLASS_COLORS = {
    (38, 23, 180):  1,   # Antenna
    (177, 132, 47): 2,   # Cable
    (129, 81, 97):  3,   # Electric pole
    (66, 132, 9):   4,   # Wind turbine
}

CLASS_NAMES = {0: "background", 1: "antenna", 2: "cable", 3: "electric_pole", 4: "wind_turbine"}

DBSCAN_PARAMS = {
    1: {"eps": 2.0, "min_samples": 10},   # Antenna
    2: {"eps": 5.0, "min_samples": 5},    # Cable
    3: {"eps": 2.0, "min_samples": 10},   # Electric pole
    4: {"eps": 5.0, "min_samples": 15},   # Wind turbine
}

CABLE_MERGE_ANGLE_DEG = 15.0
CABLE_MERGE_GAP_M = 10.0

SCENE_FILES = [f"scene_{i}.h5" for i in range(1, 11)]

# ============================================================================
# HDF5 CHUNKED READER — never loads full scene
# ============================================================================

def get_frame_boundaries(h5_path, dataset_name="lidar_points", chunk_size=2_000_000):
    """Find frame boundaries by reading in chunks — vectorized with np.diff.

    h5py with compound dtypes loads full rows, so we read in chunks of 2M rows
    (~160 MB each), extract ego columns, find transitions with np.diff, then free.
    Peak RAM: ~300 MB (1 chunk + diff arrays). Fast: numpy vectorized, no Python loop.
    """
    change_indices = []  # global indices where pose changes

    with h5py.File(h5_path, "r") as f:
        ds = f[dataset_name]
        n = ds.shape[0]
        prev_last_pose = None  # last row's pose from previous chunk

        for offset in range(0, n, chunk_size):
            end = min(offset + chunk_size, n)
            chunk = ds[offset:end]

            ex = chunk["ego_x"]
            ey = chunk["ego_y"]
            ez = chunk["ego_z"]
            eyaw = chunk["ego_yaw"]

            # Check transition between previous chunk's last row and this chunk's first
            if prev_last_pose is not None:
                cur_first = (int(ex[0]), int(ey[0]), int(ez[0]), int(eyaw[0]))
                if cur_first != prev_last_pose:
                    change_indices.append(offset)

            # Vectorized diff within chunk
            changes = np.where(
                (np.diff(ex) != 0) |
                (np.diff(ey) != 0) |
                (np.diff(ez) != 0) |
                (np.diff(eyaw) != 0)
            )[0] + 1  # +1: index of the NEW pose

            for c in changes:
                change_indices.append(offset + int(c))

            prev_last_pose = (int(ex[-1]), int(ey[-1]), int(ez[-1]), int(eyaw[-1]))

            del chunk, ex, ey, ez, eyaw
            gc.collect()

    # Build frame list: read just the first row of each frame for ego values
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


def read_frame_from_h5(h5_path, start, end, dataset_name="lidar_points"):
    """Read a single frame by row slice (~50 MB). Returns (xyz_m, r, g, b)."""
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

    xyz_m = np.column_stack((x, y, z))
    r = valid["r"].astype(np.uint8)
    g = valid["g"].astype(np.uint8)
    b = valid["b"].astype(np.uint8)

    del valid, dist_m, az_rad, el_rad, cos_el, x, y, z
    return xyz_m, r, g, b

# ============================================================================
# GT BOX PIPELINE
# ============================================================================

def map_rgb_to_class(r, g, b):
    class_ids = np.zeros(len(r), dtype=np.int64)
    for (cr, cg, cb), class_id in CLASS_COLORS.items():
        mask = (r == cr) & (g == cg) & (b == cb)
        class_ids[mask] = class_id
    return class_ids


def pca_oriented_bbox(points_m):
    center_xyz = points_m.mean(axis=0)
    centered = points_m - center_xyz
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        # Fallback: axis-aligned bbox
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
    params = DBSCAN_PARAMS[class_id]
    eps, min_samples = params["eps"], params["min_samples"]
    if len(points_m) < min_samples:
        return []

    # Subsample if too many points — DBSCAN memory is O(n²) with default algo.
    # With ball_tree it's O(n·log(n)) but still heavy for 50k+ points.
    full_points = points_m
    if len(points_m) > max_points:
        idx = np.random.choice(len(points_m), max_points, replace=False)
        points_m = points_m[idx]

    labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree").fit_predict(points_m)

    if len(full_points) > max_points:
        # Assign unsampled points to nearest cluster via simple distance
        from sklearn.neighbors import BallTree
        sampled_mask = labels >= 0
        if sampled_mask.sum() == 0:
            return []
        tree = BallTree(points_m[sampled_mask])
        _, indices = tree.query(full_points, k=1)
        full_labels = labels[sampled_mask][indices.ravel()]
        # Only keep points within eps of their assigned cluster
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
        if len(pts) < 4:  # need at least 4 points for meaningful PCA
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


def build_gt_boxes(xyz_m, r, g, b):
    class_ids = map_rgb_to_class(r, g, b)
    boxes = []
    for cid in range(1, 5):
        mask = class_ids == cid
        if mask.sum() == 0:
            continue
        clusters = cluster_class_points(xyz_m[mask], cid)
        if cid == 2 and len(clusters) > 1:
            clusters = merge_cable_clusters(clusters)
        for pts in clusters:
            if len(pts) < 3:
                continue
            bbox = pca_oriented_bbox(pts)
            boxes.append({
                "center_xyz": bbox["center_xyz"],
                "dimensions": bbox["dimensions"],
                "yaw": bbox["yaw"],
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "num_points": len(pts),
            })
    return boxes

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build GT bounding boxes from LiDAR HDF5 scenes")
    parser.add_argument("--data-dir", required=True, help="Directory containing scene_*.h5 files")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSVs")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "gt_boxes_all.csv")
    counts_path = os.path.join(output_dir, "frame_box_counts.csv")

    CSV_HEADER = ("scene,frame_idx,ego_x,ego_y,ego_z,ego_yaw,"
                  "center_x_m,center_y_m,center_z_m,"
                  "width_m,length_m,height_m,yaw_rad,"
                  "class_id,class_name,num_points\n")

    total_boxes = 0
    total_frames = 0
    t0 = time.time()

    with open(output_path, "w") as f:
        f.write(CSV_HEADER)

    counts_f = open(counts_path, "w")
    counts_f.write("scene,frame_idx,num_boxes\n")

    for scene_file in SCENE_FILES:
        h5_path = os.path.join(data_dir, scene_file)
        scene_name = scene_file.replace(".h5", "")

        if not os.path.exists(h5_path):
            print(f"  SKIP {scene_file} — not found", flush=True)
            continue

        print(f"\n[{scene_name}] Finding frame boundaries...", flush=True)
        frames_info = get_frame_boundaries(h5_path)
        n_frames = len(frames_info)
        scene_boxes = 0

        print(f"[{scene_name}] {n_frames} frames found", flush=True)

        for idx in range(n_frames):
            start, end, ego_x, ego_y, ego_z, ego_yaw = frames_info[idx]

            xyz_m, r, g, b = read_frame_from_h5(h5_path, start, end)

            if len(xyz_m) == 0:
                counts_f.write(f"{scene_name},{idx},0\n")
                total_frames += 1
                continue

            boxes = build_gt_boxes(xyz_m, r, g, b)
            counts_f.write(f"{scene_name},{idx},{len(boxes)}\n")

            if boxes:
                lines = []
                for box in boxes:
                    c, d = box["center_xyz"], box["dimensions"]
                    lines.append(
                        f"{scene_name},{idx},{ego_x},{ego_y},{ego_z},{ego_yaw},"
                        f"{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                        f"{d[0]:.4f},{d[1]:.4f},{d[2]:.4f},{box['yaw']:.4f},"
                        f"{box['class_id']},{box['class_name']},{box['num_points']}\n"
                    )
                with open(output_path, "a") as f:
                    f.writelines(lines)
                scene_boxes += len(boxes)

            del xyz_m, r, g, b, boxes
            total_frames += 1

            # Progress every 10 frames
            if (idx + 1) % 10 == 0:
                print(f"  [{scene_name}] {idx+1}/{n_frames} frames, {scene_boxes} boxes so far", flush=True)

        total_boxes += scene_boxes
        print(f"  [{scene_name}] DONE — {scene_boxes} boxes from {n_frames} frames", flush=True)

        del frames_info
        gc.collect()

    counts_f.close()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_boxes} GT boxes across {total_frames} frames")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {output_path}")
    print(f"Counts: {counts_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
