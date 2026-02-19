"""
Ground truth bounding box reconstruction from point-wise RGB labels.

Pipeline per frame:
  1. RGB → class_id (exact color match)
  2. Per-class DBSCAN clustering → object instances
  3. PCA → oriented bounding box (center, dims, yaw)

Output format: List[dict] with keys:
  center_xyz (3,), dimensions (3,), yaw, class_id
"""

import numpy as np
from sklearn.cluster import DBSCAN

from config import CLASS_COLORS, CLASS_NAMES, DBSCAN_PARAMS, CABLE_MERGE_ANGLE_DEG, CABLE_MERGE_GAP_M

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def map_rgb_to_class(r, g, b):
    """Map RGB arrays to class ID array. 0 = background.

    Args:
        r, g, b: numpy arrays of uint8 RGB values

    Returns:
        numpy array of int64 class IDs, shape (N,)
    """
    class_ids = np.zeros(len(r), dtype=np.int64)
    for (cr, cg, cb), class_id in CLASS_COLORS.items():
        mask = (r == cr) & (g == cg) & (b == cb)
        class_ids[mask] = class_id
    return class_ids


def pca_oriented_bbox(points_m):
    """Compute an oriented 3D bounding box using PCA.

    Args:
        points_m: numpy array (N, 3) in meters [x_m, y_m, z_m]

    Returns:
        dict with:
          center_xyz: (3,) center of the box in meters
          dimensions: (3,) [width, length, height] in meters
          yaw: float, rotation angle in radians (XY plane)
          eigenvectors: (3, 3) PCA axes for debugging
    """
    center_xyz = points_m.mean(axis=0)

    # PCA on centered points
    centered = points_m - center_xyz
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue descending (largest variance first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Project points onto principal axes
    projected = centered @ eigenvectors  # (N, 3)

    # Dimensions = range along each axis
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    dimensions = maxs - mins  # [width, length, height] along PCA axes

    # Recenter: PCA center might not be the box center if points are asymmetric
    box_center_pca = (mins + maxs) / 2.0
    center_xyz = center_xyz + eigenvectors @ box_center_pca

    # Yaw = angle of first principal axis projected onto XY plane
    axis1_xy = eigenvectors[:2, 0]  # first eigenvector, XY components
    yaw = np.arctan2(axis1_xy[1], axis1_xy[0])

    return {
        "center_xyz": center_xyz,
        "dimensions": dimensions,
        "yaw": float(yaw),
        "eigenvectors": eigenvectors,
    }


def cluster_class_points(points_m, class_id):
    """Cluster points of a single class into object instances using DBSCAN.

    Args:
        points_m: numpy array (N, 3) in meters
        class_id: int, class ID (1-4)

    Returns:
        list of numpy arrays, each containing points of one instance
    """
    params = DBSCAN_PARAMS[class_id]
    eps = params["eps"]
    min_samples = params["min_samples"]

    if len(points_m) < min_samples:
        logger.warning(f"Class {CLASS_NAMES[class_id]}: only {len(points_m)} points, "
                       f"need {min_samples} for DBSCAN. Skipping.")
        return []

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_m)

    clusters = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise label

    for label in sorted(unique_labels):
        cluster_points = points_m[labels == label]
        clusters.append(cluster_points)

    n_noise = (labels == -1).sum()
    if n_noise > 0:
        logger.info(f"Class {CLASS_NAMES[class_id]}: {len(clusters)} clusters, "
                     f"{n_noise} noise points discarded")

    return clusters


def merge_cable_clusters(clusters):
    """Merge co-linear cable clusters that are close together.

    Two cable clusters are merged if:
      - Their PCA axis 1 directions are within CABLE_MERGE_ANGLE_DEG
      - The gap between them is within CABLE_MERGE_GAP_M

    Args:
        clusters: list of numpy arrays (N, 3) — cable point clusters

    Returns:
        list of numpy arrays — merged clusters
    """
    if len(clusters) <= 1:
        return clusters

    angle_thresh_rad = np.radians(CABLE_MERGE_ANGLE_DEG)
    gap_thresh_m = CABLE_MERGE_GAP_M

    # Compute PCA axis 1 and center for each cluster
    cluster_info = []
    for pts in clusters:
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        axis1 = eigenvectors[:, order[0]]
        # Normalize to positive x (consistent direction)
        if axis1[0] < 0:
            axis1 = -axis1
        cluster_info.append({
            "points": pts,
            "center": pts.mean(axis=0),
            "axis1": axis1,
        })

    # Greedy merging
    merged = [False] * len(cluster_info)
    result = []

    for i in range(len(cluster_info)):
        if merged[i]:
            continue

        current_pts = cluster_info[i]["points"]

        for j in range(i + 1, len(cluster_info)):
            if merged[j]:
                continue

            # Check angle between axes
            dot = abs(np.dot(cluster_info[i]["axis1"], cluster_info[j]["axis1"]))
            dot = min(dot, 1.0)  # numerical safety
            angle = np.arccos(dot)

            if angle > angle_thresh_rad:
                continue

            # Check gap between clusters (min distance between any two points)
            # Approximate with center-to-center distance minus half-extents
            center_dist = np.linalg.norm(
                cluster_info[i]["center"] - cluster_info[j]["center"]
            )
            # Rough half-extent along axis 1
            extent_i = np.abs(
                (cluster_info[i]["points"] - cluster_info[i]["center"]) @ cluster_info[i]["axis1"]
            ).max()
            extent_j = np.abs(
                (cluster_info[j]["points"] - cluster_info[j]["center"]) @ cluster_info[j]["axis1"]
            ).max()
            gap = center_dist - extent_i - extent_j

            if gap <= gap_thresh_m:
                # Merge
                current_pts = np.vstack([current_pts, cluster_info[j]["points"]])
                merged[j] = True
                logger.info(f"Cable: merged clusters {i} and {j} (angle={np.degrees(angle):.1f}°, gap={gap:.1f}m)")

        result.append(current_pts)

    return result


def build_gt_boxes_for_frame(xyz_m, r, g, b):
    """Build ground truth bounding boxes for a single frame.

    Args:
        xyz_m: numpy array (N, 3) in meters [x_m, y_m, z_m]
        r, g, b: numpy arrays (N,) of uint8 RGB labels

    Returns:
        list of dict, each with:
          center_xyz: (3,) in meters
          dimensions: (3,) [width, length, height] in meters
          yaw: float in radians
          class_id: int (1-4)
          class_name: str
          num_points: int
    """
    class_ids = map_rgb_to_class(r, g, b)
    boxes = []

    for cid in range(1, 5):  # classes 1-4
        mask = class_ids == cid
        if mask.sum() == 0:
            continue

        class_points_m = xyz_m[mask]

        # Cluster into instances
        clusters = cluster_class_points(class_points_m, cid)

        # Cable-specific: merge co-linear clusters
        if cid == 2 and len(clusters) > 1:
            clusters = merge_cable_clusters(clusters)

        # Build box for each cluster
        for cluster_pts in clusters:
            if len(cluster_pts) < 3:
                continue  # need at least 3 points for PCA

            bbox = pca_oriented_bbox(cluster_pts)
            boxes.append({
                "center_xyz": bbox["center_xyz"],
                "dimensions": bbox["dimensions"],
                "yaw": bbox["yaw"],
                "class_id": cid,
                "class_name": CLASS_NAMES[cid],
                "num_points": len(cluster_pts),
            })

    return boxes


def boxes_to_csv_rows(boxes, ego_x, ego_y, ego_z, ego_yaw):
    """Convert boxes to CSV-ready rows.

    Args:
        boxes: list of dict from build_gt_boxes_for_frame
        ego_x, ego_y, ego_z: int, raw ego position (centimeters)
        ego_yaw: int, raw ego yaw (centièmes de degré)

    Returns:
        list of dict with CSV column names
    """
    from config import CLASS_NAMES_CSV

    rows = []
    for box in boxes:
        rows.append({
            "ego_x": int(ego_x),
            "ego_y": int(ego_y),
            "ego_z": int(ego_z),
            "ego_yaw": int(ego_yaw),
            "bbox_center_x": f"{box['center_xyz'][0]:.4f}",
            "bbox_center_y": f"{box['center_xyz'][1]:.4f}",
            "bbox_center_z": f"{box['center_xyz'][2]:.4f}",
            "bbox_width": f"{box['dimensions'][0]:.4f}",
            "bbox_length": f"{box['dimensions'][1]:.4f}",
            "bbox_height": f"{box['dimensions'][2]:.4f}",
            "bbox_yaw": f"{box['yaw']:.4f}",
            "class_ID": box["class_id"],
            "class_label": CLASS_NAMES_CSV[box["class_id"]],
        })
    return rows
