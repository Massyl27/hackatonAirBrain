"""
PyTorch Dataset for LiDAR point cloud segmentation.
Reads HDF5 files frame-by-frame (memory-safe), returns fixed-size point samples.
"""

import gc
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Class RGB → ID mapping (inline to keep script standalone)
CLASS_COLORS = {
    (38, 23, 180): 1,    # Antenna
    (177, 132, 47): 2,   # Cable
    (129, 81, 97): 3,    # Electric pole
    (66, 132, 9): 4,     # Wind turbine
}


def get_frame_boundaries(h5_path, dataset_name="lidar_points", chunk_size=2_000_000):
    """Find frame boundaries by reading in chunks — vectorized."""
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


def map_rgb_to_class(r, g, b):
    """Map RGB to class IDs. 0 = background."""
    class_ids = np.zeros(len(r), dtype=np.int64)
    for (cr, cg, cb), cid in CLASS_COLORS.items():
        mask = (r == cr) & (g == cg) & (b == cb)
        class_ids[mask] = cid
    return class_ids


class LidarSegDataset(Dataset):
    """PyTorch Dataset that reads LiDAR frames from HDF5 on-the-fly.

    Each __getitem__ returns:
        points: (num_points, 4) float32 — [x, y, z, reflectivity_normalized]
        labels: (num_points,) int64 — class IDs [0..4]
        meta: dict with scene, frame_idx, ego pose
    """

    def __init__(self, data_dir, scene_files, num_points=32000,
                 augment=False, jitter_std=0.02):
        self.data_dir = data_dir
        self.num_points = num_points
        self.augment = augment
        self.jitter_std = jitter_std

        # Build index: list of (h5_path, start, end, ego_x, ego_y, ego_z, ego_yaw, scene_name, frame_idx)
        self.index = []
        for sf in scene_files:
            h5_path = os.path.join(data_dir, sf)
            scene_name = sf.replace(".h5", "")
            if not os.path.exists(h5_path):
                continue
            frames = get_frame_boundaries(h5_path)
            for idx, (start, end, ex, ey, ez, eyaw) in enumerate(frames):
                self.index.append((h5_path, start, end, ex, ey, ez, eyaw, scene_name, idx))

        print(f"LidarSegDataset: {len(self.index)} frames from {len(scene_files)} scenes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        h5_path, start, end, ex, ey, ez, eyaw, scene_name, frame_idx = self.index[i]

        # Read frame from HDF5
        with h5py.File(h5_path, "r") as f:
            chunk = f["lidar_points"][start:end]

        # Filter valid points
        valid = chunk[chunk["distance_cm"] > 0]
        del chunk

        # Spherical → Cartesian
        dist_m = valid["distance_cm"].astype(np.float64) / 100.0
        az_rad = np.radians(valid["azimuth_raw"].astype(np.float64) / 100.0)
        el_rad = np.radians(valid["elevation_raw"].astype(np.float64) / 100.0)

        cos_el = np.cos(el_rad)
        x = dist_m * cos_el * np.cos(az_rad)
        y = -dist_m * cos_el * np.sin(az_rad)
        z = dist_m * np.sin(el_rad)

        xyz = np.column_stack((x, y, z)).astype(np.float32)
        reflectivity = (valid["reflectivity"].astype(np.float32) / 255.0).reshape(-1, 1)
        labels = map_rgb_to_class(
            valid["r"].astype(np.uint8),
            valid["g"].astype(np.uint8),
            valid["b"].astype(np.uint8),
        )

        del valid, dist_m, az_rad, el_rad, cos_el, x, y, z

        n_pts = len(xyz)

        # Handle empty frames (all points invalid)
        if n_pts == 0:
            features = np.zeros((self.num_points, 4), dtype=np.float32)
            labels = np.zeros(self.num_points, dtype=np.int64)
            return torch.from_numpy(features), torch.from_numpy(labels)

        # Fixed-size sampling
        if n_pts >= self.num_points:
            idx = np.random.choice(n_pts, self.num_points, replace=False)
        else:
            idx = np.random.choice(n_pts, self.num_points, replace=True)

        xyz = xyz[idx]
        reflectivity = reflectivity[idx]
        labels = labels[idx]

        # Augmentation
        if self.augment:
            # Random rotation around Z
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t, 0],
                            [sin_t, cos_t, 0],
                            [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ rot.T

            # Jitter
            xyz += np.random.normal(0, self.jitter_std, xyz.shape).astype(np.float32)

            # Random point drop (replace 10-50% with duplicates of remaining)
            drop_ratio = np.random.uniform(0.0, 0.25)
            n_drop = int(self.num_points * drop_ratio)
            if n_drop > 0:
                keep_idx = np.random.choice(self.num_points, self.num_points - n_drop, replace=False)
                fill_idx = np.random.choice(keep_idx, n_drop, replace=True)
                all_idx = np.concatenate([keep_idx, fill_idx])
                np.random.shuffle(all_idx)
                xyz = xyz[all_idx]
                reflectivity = reflectivity[all_idx]
                labels = labels[all_idx]

        # Concatenate features: [x, y, z, reflectivity]
        features = np.concatenate([xyz, reflectivity], axis=1)

        return (
            torch.from_numpy(features),   # (num_points, 4)
            torch.from_numpy(labels),      # (num_points,)
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    scene_files = [f"scene_{i}.h5" for i in range(1, 11)]
    ds = LidarSegDataset(args.data_dir, scene_files, num_points=32000, augment=True)
    print(f"\nTotal frames: {len(ds)}")

    features, labels = ds[0]
    print(f"features: {features.shape}, labels: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels, minlength=5).tolist()}")
