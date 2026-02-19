# Airbus AI Hackathon 2026 — CLAUDE.md

## Projet

**Défi :** Détection et classification d'obstacles 3D en temps réel à partir de nuages de points LiDAR pour hélicoptères Airbus (prévention de collision avec pylônes, câbles, antennes, éoliennes).

**Deadline :** J-7 — les fichiers d'évaluation arrivent entre 00h01 et 01h00 AM le dernier jour.

---

## BMAD — Configuration

- **Framework :** BMAD v6.0.1 installé dans `_bmad/`
- **Outputs :** `_bmad-output/`
- **Langue de communication :** Français
- **Langue des documents :** Anglais
- **Utilisateur :** Quentin

Pour activer un agent BMAD, référencer le fichier agent correspondant dans `_bmad/bmm/agents/` ou `_bmad/core/agents/`.

---

## Environnement d'exécution

**Pas de GPU local — tout le code ML tourne sur Google Colab.**

- Les données (1.5 Go, `airbus_hackathon_trainingdata.zip`) sont uploadées sur **Google Drive**
- Les notebooks Colab montent le Drive via `drive.mount('/content/drive')`
- GPU cible : T4 (gratuit) ou A100 (Colab Pro)
- Format de sauvegarde modèle : **ONNX** (préféré) ou PyTorch `.pt`

---

## Données

### Structure des fichiers
- **10 fichiers HDF5** d'entraînement : `scene_1.h5` … `scene_10.h5`
- Dataset name dans le HDF5 : `lidar_points`
- Chaque fichier contient plusieurs frames (10 frames par scène = 100 frames totales)

### Champs disponibles
| Champ | Description | Unité |
|-------|-------------|-------|
| `distance_cm` | Distance LiDAR → point | Centimètres |
| `azimuth_raw` | Angle horizontal | Centièmes de degré |
| `elevation_raw` | Angle vertical | Centièmes de degré |
| `reflectivity` | Intensité du retour laser | 8-bit (0-255) |
| `r, g, b` | Label de classe (couleur ground truth) | 3x 8-bit |
| `ego_x, ego_y, ego_z` | Position du véhicule | Centimètres |
| `ego_yaw` | Yaw du véhicule | Centièmes de degré |

### Identification d'une frame
Le quadruplet `(ego_x, ego_y, ego_z, ego_yaw)` identifie une frame unique.

### Labels RGB → Classes
| Class ID | Label | R | G | B |
|----------|-------|---|---|---|
| 0 | Antenna | 38 | 23 | 180 |
| 1 | Cable | 177 | 132 | 47 |
| 2 | Electric pole | 129 | 81 | 97 |
| 3 | Wind turbine | 66 | 132 | 9 |

> Les points non labellisés (background, sol, arbres...) ont des couleurs différentes.

---

## Toolkit fourni

Fichiers dans `airbus_hackathon_toolkit/` :

- `lidar_utils.py` — Fonctions utilitaires :
  - `load_h5_data(file_path)` → DataFrame pandas
  - `get_unique_poses(df)` → liste des frames uniques
  - `filter_by_pose(df, pose_row)` → points d'une frame
  - `spherical_to_local_cartesian(df)` → coordonnées XYZ locales (mètres)
- `visualize.py` — Visualisation Open3D avec `--file` et `--pose-index`

### Convention de coordonnées
- **Repère local LiDAR** (left-handed, Z up)
- Conversion : `x = d·cos(el)·cos(az)`, `y = -d·cos(el)·sin(az)`, `z = d·sin(el)`

---

## Sorties attendues (livrables)

### CSVs de prédiction
Pour chaque frame, une ligne par objet détecté :

```
ego_x, ego_y, ego_z, ego_yaw,
bbox_center_x, bbox_center_y, bbox_center_z,
bbox_width, bbox_length, bbox_height,
bbox_yaw,
class_ID, class_label
```

- Coordonnées du centre en **mètres**, origine = position LiDAR
- 8 fichiers d'évaluation (2 scènes × 4 densités : 100%, 75%, 50%, 25%)

### Modèle
- Format **ONNX** ou PyTorch
- Inclure le **nombre de paramètres** dans la présentation

---

## Critères d'évaluation (par ordre d'importance)

1. **mAP @ IoU=0.5** — précision de détection principale
2. **Mean IoU (bonne classe)** — qualité des bounding boxes
3. **Robustesse** — maintenir les perf à 25% des points
4. **Efficience** — peu de paramètres
5. **Stabilité** — cohérence entre scène connue et inconnue

---

## Architecture technique cible

### Pipeline ML
```
HDF5 → Spherical→Cartesian → Voxelization/Pillars → Backbone 3D → Detection Head → Bounding Boxes
```

### Stack Python
- `h5py`, `numpy`, `pandas` — chargement données
- `torch` (PyTorch) — entraînement
- `open3d` — visualisation nuages de points
- `scikit-learn` — clustering (DBSCAN pour reconstruction des GT boxes)
- `onnx`, `onnxruntime` — export et inférence

### Approche recommandée
- **PointPillars** (architecture légère, rapide, peu de paramètres) — idéal pour le critère efficience
- Alternative : clustering DBSCAN + features géométriques + classifieur léger

---

## Règles de développement

- **JAMAIS de commit/push sans accord explicite** de Quentin
- Tout le code d'entraînement doit tourner dans un **notebook Colab** (`*.ipynb`)
- Le code d'inférence doit être un **script Python standalone** (`inference.py`)
- Inclure `requirements.txt` séparé pour train et pour inférence
- Les chemins de fichiers dans les notebooks doivent pointer vers Google Drive
- Sauvegarder les checkpoints régulièrement sur Drive (pas de perte si Colab déconnecte)

---

## Structure du projet

```
airbus_hackathon/
├── CLAUDE.md                        ← ce fichier
├── airbus_hackathon_toolkit/        ← toolkit fourni par Airbus
│   ├── lidar_utils.py
│   ├── visualize.py
│   └── requirements.txt
├── airbus_hackathon_trainingdata.zip ← données d'entraînement (1.5 Go)
├── notebooks/                       ← notebooks Colab
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_inference.ipynb
├── src/                             ← code source Python
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── outputs/                         ← CSVs de prédiction générés
├── _bmad/                           ← BMAD framework
└── _bmad-output/                    ← documents générés par BMAD
```

---

## Contact Airbus
- Email hackathon : contact.hackathon-ai.ah@airbus.com
