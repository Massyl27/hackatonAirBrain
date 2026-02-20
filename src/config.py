"""
Central configuration for the Airbus LiDAR obstacle detection pipeline.
Single source of truth for all constants, class labels, and hyperparameters.
"""

# =============================================================================
# CLASS DEFINITIONS — RGB → Class ID mapping
# =============================================================================

# Ground truth RGB colors → class ID (exact match from Airbus spec)
CLASS_COLORS = {
    (38, 23, 180):  1,   # Antenna
    (177, 132, 47): 2,   # Cable
    (129, 81, 97):  3,   # Electric pole
    (66, 132, 9):   4,   # Wind turbine
}

# Class ID → internal name (snake_case, for code)
CLASS_NAMES = {
    0: "background",
    1: "antenna",
    2: "cable",
    3: "electric_pole",
    4: "wind_turbine",
}

# Class ID → CSV label (exact Airbus spec strings, capitalization matters)
CLASS_NAMES_CSV = {
    1: "Antenna",
    2: "Cable",
    3: "Electric Pole",
    4: "Wind Turbine",
}

# Internal class_id (1-4) → Airbus class_ID (0-3)
CLASS_ID_TO_AIRBUS = {1: 0, 2: 1, 3: 2, 4: 3}

NUM_CLASSES = 5  # including background (class 0)

# =============================================================================
# DBSCAN CLUSTERING PARAMETERS (per class)
# Initial estimates — to be tuned after data exploration (Story 1.3)
# =============================================================================

DBSCAN_PARAMS = {
    1: {"eps": 2.0, "min_samples": 10},   # Antenna — compact vertical
    2: {"eps": 5.0, "min_samples": 5},    # Cable — sparse linear
    3: {"eps": 2.0, "min_samples": 10},   # Electric pole — compact vertical
    4: {"eps": 5.0, "min_samples": 15},   # Wind turbine — large object
}

# Cable merging thresholds
CABLE_MERGE_ANGLE_DEG = 15.0   # max angle between PCA axes to merge
CABLE_MERGE_GAP_M = 10.0       # max gap between clusters to merge

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAIN_CONFIG = {
    "num_points": 32000,        # subsample per frame during training
    "batch_size": 4,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "val_scene": "scene_8",     # most diverse scene (4 classes, highest entropy) — Story 1.2
    # Loss weights calibrated from actual class frequencies (Story 1.2)
    "class_weights": [0.1, 14.14, 70.41, 49.94, 6.29],  # [bg, antenna, cable, pole, turbine]
}

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

AUGMENTATION = {
    "rotation_z": True,             # random rotation around Z axis [0, 2π]
    "jitter_std": 0.02,             # Gaussian noise σ in meters
    "drop_ratio_range": (0.1, 0.5), # random point drop ratio range
}

# =============================================================================
# PATHS (Google Drive — for Colab notebooks)
# =============================================================================

DRIVE_BASE = "/content/drive/MyDrive/airbus_hackathon"
DRIVE_DATA = f"{DRIVE_BASE}/data"
DRIVE_CHECKPOINTS = f"{DRIVE_BASE}/checkpoints"
DRIVE_OUTPUTS = f"{DRIVE_BASE}/outputs"

# Scene files
SCENE_FILES = [f"scene_{i}.h5" for i in range(1, 11)]
