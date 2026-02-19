# Suivi du Projet — Airbus AI Hackathon 2026

## Journal de bord (pour le rapport final)

---

### 2026-02-18 — Soirée : Lancement du projet

#### Story 1.1 — Data Exploration & Coordinate Validation
**Résultats clés :**
- **998 frames** au total (et non 100 comme anticipé dans le brief) — 10x plus de données
- **575,000 points par frame** (constant), sauf scene_10 : 586,735
- 10 scènes × ~100 frames/scène
- Colonnes confirmées : `distance_cm, azimuth_raw, elevation_raw, reflectivity, r, g, b, ego_x, ego_y, ego_z, ego_yaw`
- Conversion sphérique → cartésien validée visuellement (pas de mirroring, distances cohérentes)
- Réflectivité : range [8, 255], dtype uint16, concentrée sur les basses valeurs

**Observation :** Les frames de test du notebook 01 ne montraient que des antennes → besoin d'analyser toutes les scènes.

#### Story 1.2 — Class Distribution Analysis
**Résultats clés :**

| Classe | Points totaux | % obstacles | Observation |
|--------|--------------|-------------|-------------|
| Wind turbine | 4,050,806 | 60.2% | Classe dominante |
| Antenna | 1,802,148 | 26.8% | Bien représentée |
| Electric pole | 510,143 | 7.6% | Rare |
| Cable | 361,878 | 5.4% | La plus rare — confirme que c'est le défi principal |

**Heatmap par scène :**
- Les 4 classes présentes dans presque toutes les scènes (sauf scene_5 : 0 electric_pole)
- scene_8 = la plus équilibrée (20% antenna, 25% cable, 5% pole, 49% turbine)
- scene_9 = dominée par éoliennes (84%)

**Décisions prises :**
- **Scène de validation : scene_8** (diversity score le plus élevé : 4.717)
- **Loss weights calibrés :** `[0.10, 14.14, 70.41, 49.94, 6.29]` (bg, antenna, cable, pole, turbine)
- Les poids sont bien plus extrêmes que les estimations initiales (cable 70 vs 10, pole 50 vs 3)

**Fichiers produits :**
- `src/config.py` — mis à jour avec les vrais poids et scene_8
- `outputs/frame_stats.csv` — stats par frame sauvegardées sur Drive

---

### 2026-02-19 — Story 1.3/1.4 : GT Box Reconstruction

#### Problème rencontré : OOM sur Colab (12 Go RAM) — 3 crashs successifs

| Tentative | Approche | RAM pic estimé | Résultat |
|-----------|----------|---------------|----------|
| v1 | pandas `load_h5_data()` + `del df; gc.collect()` | ~8-10 Go | OOM |
| v2 | numpy structured array `h5py` direct + `del; gc` | ~5-6 Go | OOM |
| v3 | **Lecture chunked frame-par-frame** (slice h5py) | **~200 Mo** | En test |

**Analyse du problème :**
- Chaque scène HDF5 = 57.5M points. En pandas DataFrame : ~4-5 Go. En numpy structured array : ~1.5-2 Go.
- Même le numpy structured array (v2) crashe car : les opérations intermédiaires (`np.column_stack`, `np.unique`, masking booléen sur 57M points) créent des copies temporaires. Cumulé avec DBSCAN (qui alloue des matrices de distances) et matplotlib (qui garde les figures), le pic dépasse 12 Go.

**Fix définitif (v3)** : Ne **jamais** charger une scène entière.
- `get_frame_boundaries()` : lit **uniquement les 4 colonnes ego** depuis le HDF5 (~30 Mo) pour identifier les limites de chaque frame (transitions de pose), puis les libère
- `read_frame_from_h5()` : lit un **slice** `ds[start:end]` pour **une seule frame** (~50 Mo pour 575k points)
- `plt.close(fig)` + `matplotlib.use('Agg')` pour libérer la RAM des figures
- Peak RAM : ~200 Mo max (1 frame + DBSCAN + 1 figure)

**Leçon retenue :** Sur Colab (12 Go RAM), avec des fichiers HDF5 de 57M+ lignes, même `h5py[:]` (chargement complet en numpy) peut crasher quand on enchaîne les opérations. La seule approche fiable est le **slicing h5py frame par frame** — on ne charge jamais plus d'1 frame en mémoire.

#### Pipeline GT Box
- **Méthode** : RGB → class_id (exact color match) → DBSCAN clustering par classe → PCA oriented bounding box
- **DBSCAN params** : eps/min_samples ajustés par classe (antenna: 2.0/10, cable: 5.0/5, pole: 2.0/10, turbine: 5.0/15)
- **Cable merging** : Fusion des clusters co-linéaires (angle < 15°, gap < 10m)
- **Sortie** : `gt_boxes_all.csv` avec center_xyz, dimensions, yaw, class_id pour chaque objet de chaque frame

**Fichiers produits :**
- `src/gt_boxes.py` — module Python réutilisable pour la reconstruction de GT boxes
- `scripts/build_gt_boxes.py` — script standalone (exécuté sur PC Linux via SSH)
- `outputs/gt_boxes_all.csv` — 5,839 GT boxes
- `outputs/frame_box_counts.csv` — comptage par frame

#### Résultats — GT Box Reconstruction

**Exécuté sur PC Linux** (14 Go RAM) via SSH — **2.7 minutes** pour 1,000 frames.

| Classe | Boxes | % | Points médians | Dimension médiane (W×L×H) |
|--------|-------|---|----------------|---------------------------|
| antenna | 1,963 | 33.6% | 44 | 7.7 × 3.0 × 1.8 m |
| cable | 1,661 | 28.4% | 13 | 14.8 × 4.3 × 0.16 m |
| wind_turbine | 1,250 | 21.4% | 292 | 23.8 × 6.6 × 1.2 m |
| electric_pole | 965 | 16.5% | 21 | 5.2 × 2.5 × 1.3 m |

- **906/1000 frames** contiennent au moins 1 obstacle (90.6%)
- Moyenne de **6.4 boxes** par frame (médiane : 4)
- Frame la plus dense : 50 boxes (scene_3)

**Observations clés pour le modèle :**
- **Câbles = défi principal** : 41% ont < 10 points LiDAR, hauteur quasi-nulle (0.16m)
- **Hauteur systématiquement sous-estimée** pour toutes les classes (géométrie hélicoptère = vue latérale)
- **Éoliennes = cibles les plus riches** (292 pts médians, grandes surfaces réfléchissantes)
- Scene_5 : 0 electric_pole — variation géographique réelle
- Scene_3 : outlier avec 1,507 boxes (zone dense, corridors de câbles)
- Quelques cas d'over-merging DBSCAN sur les câbles (box de 300m — probablement 2 câbles fusionnés)

---

### 2026-02-19 — Story 2.1-2.3 : PointNet-lite Training (v1 → v2)

#### v1 — Premier entraînement (résultats décevants)

**Setup :**
- Modèle : PointNet-lite, **116,549 paramètres**
- Backbone : 64→128→256 + global feature
- Loss : Weighted CrossEntropy (`[0.1, 14.14, 70.41, 49.94, 6.29]`)
- LR : 1e-3, cosine annealing, 50 epochs, batch=8
- Train : 900 frames (9 scènes), Val : 100 frames (scene_8)
- GPU : Tesla T4, temps total : **36.7 min**

**Résultats v1 :**

| Classe | IoU |
|--------|-----|
| background | 0.8553 |
| antenna | 0.0051 |
| cable | 0.0047 |
| electric_pole | 0.0054 |
| wind_turbine | 0.1994 |
| **mIoU (all)** | **0.2140** |

**Diagnostic :** Le modèle prédit quasi-tout en background. Le weighted CE ne suffit pas face au déséquilibre extrême (~90% de background). Le best mIoU est atteint à l'epoch 6 puis dégénère — instabilité d'entraînement.

**Bugs rencontrés sur Colab :**
- `torch.cuda.get_device_properties(0).total_mem` → `AttributeError` (corrigé : `total_memory`)
- `torch.load()` → `UnpicklingError` avec PyTorch 2.6 (corrigé : `weights_only=False`)
- `torch.onnx.export` → `ModuleNotFoundError: onnxscript` (corrigé : `pip install onnx onnxscript`)
- Cache notebook Colab → renommage avec timestamp pour forcer le rechargement

**Fichiers produits :**
- `notebooks/04_training_20260219.ipynb` — notebook v1 (corrigé)
- `checkpoints/best_model.pt` — meilleur modèle v1 (epoch 6, mIoU=0.2140)
- `checkpoints/pointnet_lite.onnx` — export ONNX (0.02 MB)

#### v2 — Améliorations en cours

**Changements v1 → v2 :**

| Aspect | v1 | v2 |
|--------|----|----|
| Loss | Weighted CE | **Focal Loss** (gamma=2.0) |
| Métrique | mIoU (avec BG) | **Obstacle-only mIoU** |
| LR | 1e-3, cosine | **3e-4**, warmup 5 epochs + cosine |
| Grad clipping | Non | **max_norm=1.0** |
| Backbone | 64→128→256 | **64→128→256→512** |
| Skip connections | global seulement | **Multi-scale** (e1+e2+e3+global) |
| Epochs | 50 | **80** |
| Augmentation | rotation+jitter+drop | + **random scale** (0.9-1.1) |
| Checkpoints | `checkpoints/` | `checkpoints_v2/` (séparé) |

**Justifications :**
- **Focal Loss** : réduit la contribution des exemples faciles (background) et force le modèle à apprendre les classes rares. Alpha pondéré : `[0.02, 0.20, 0.40, 0.25, 0.13]` (cable le plus pondéré)
- **Obstacle-only mIoU** : le background IoU de 0.85 masquait la mauvaise performance sur les obstacles
- **LR warmup** : évite l'instabilité initiale qui causait le pic-puis-effondrement en v1
- **Multi-scale skip** : les features locales (e1, e2) donnent la géométrie fine, le global feature donne le contexte

**Fichier :** `notebooks/04_training_v2.ipynb`

#### v2 — Résultats (Focal Loss seule, sans balancing)

**Résultat :** Pire que v1 — obstacle mIoU ~0.027. Le modèle est encore plus collé au background (BG IoU = 0.977). La Focal Loss seule ne résout pas le déséquilibre quand >90% des points sont background.

**Fichier :** `notebooks/04_training_v2.ipynb`, `checkpoints_v2/`

---

### 2026-02-19 ~22h — Story 2.1-2.3 : PointNet-lite v3 (balanced sampling)

#### Le fix clé : balanced sampling

**Problème identifié :** Avec un échantillonnage aléatoire de 32k points sur 575k, seulement ~1,700 sont des obstacles (5%). Le modèle n'a presque rien à apprendre.

**Solution :** Forcer un ratio 50% obstacle / 50% background dans chaque sample. Le modèle voit maintenant **~32k points obstacles par sample** (20x plus qu'avant).

**Changements v2 → v3 :**

| Aspect | v2 | v3 |
|--------|----|----|
| Sampling | Random (5% obstacles) | **Balanced 50/50** |
| Points | 32k | **64k** |
| Features | 4 (xyz+refl) | **5** (+distance normalisée) |
| Batch | 8 | 8 |
| GPU | T4 (15 GB) | **A100 (42.4 GB)** |
| VRAM utilisé | ~2 GB | **11.2 GB (26%)** |

**Résultats v3 (best epoch 54) :**

| Classe | v1 | v2 | **v3** |
|--------|-----|-----|--------|
| background | 0.855 | 0.977 | **0.787** |
| antenna | 0.005 | 0.031 | **0.290** |
| cable | 0.005 | 0.008 | **0.283** |
| electric_pole | 0.005 | 0.031 | 0.041 |
| wind_turbine | 0.199 | 0.036 | 0.059 |
| **obstacle mIoU** | ~0.05 | 0.027 | **0.168** |

- **454,213 paramètres**, ONNX 0.02 MB
- 80 epochs, 67.3 min sur A100
- La courbe montait encore à epoch 80 → modèle n'avait pas fini d'apprendre

**Analyse :**
- antenna et cable : **x50 d'amélioration** vs v1 — le balanced sampling fonctionne
- electric_pole (0.04) et wind_turbine (0.06) restent faibles — objets spatialement grands, points dispersés
- Le background IoU baisse (0.79 vs 0.98) = le modèle prédit maintenant des obstacles (bon signe)

**Fichiers produits :**
- `notebooks/04_training_v3.ipynb`
- `checkpoints_v3/best_model_v3.pt` (epoch 54, obs mIoU=0.168)
- `checkpoints_v3/pointnet_lite_v3.onnx` (0.02 MB)
- `checkpoints_v3/training_curves_v3.png`

---

### 2026-02-19 ~23h30 — Story 2.1-2.3 : PointNet v4 (class-balanced + A100)

#### Changements v3 → v4

| Aspect | v3 | v4 |
|--------|----|----|
| Sampling obstacles | 50/50 (bg/obs global) | **Class-balanced** (equal per class) |
| Modèle | 454k params | **1,882,693 params** |
| Encoder | 64→128→256→512 | **64→128→256→512→1024** |
| Decoder | 3 layers | **4 layers** (512→256→128→5) |
| Batch | 8 | **16** |
| Epochs | 80 | **150** |
| LR schedule | Cosine simple | **Cosine warm restart** (cycle=50) |
| GPU | A100 40GB | **A100 80GB** |
| VRAM | 11.2 GB | **46.0 GB (54%)** |

#### Résultats v4 (best epoch 132)

| Classe | v1 | v3 | **v4** | Tendance |
|--------|-----|------|--------|----------|
| background | 0.855 | 0.787 | **0.768** | — |
| antenna | 0.005 | 0.290 | **0.256** | stable |
| cable | 0.005 | 0.283 | **0.378** | +33% |
| electric_pole | 0.005 | 0.041 | **0.017** | en difficulté |
| wind_turbine | 0.199 | 0.059 | **0.170** | +188% |
| **obstacle mIoU** | 0.05 | 0.168 | **0.205** | +22% |

- **1,882,693 paramètres**, ONNX 0.03 MB
- 150 epochs, 131.6 min sur A100 80GB
- Warm restarts visibles aux epochs 55 et 105 (sauts de mIoU)
- La courbe montait encore → itération possible avec plus d'epochs

**Analyse :**
- **cable** = meilleure classe obstacle (0.378) — le class-balanced sampling l'a bien boosté
- **wind_turbine** = grosse amélioration vs v3 (0.06→0.17) grâce à l'allocation égale de points
- **electric_pole** = toujours le point faible (0.017). Seulement 21 pts médians/objet, absent de scene_5. Le modèle le confond probablement avec background ou antenna.

**Fichiers produits :**
- `notebooks/04_training_v4.ipynb`
- `checkpoints_v4/best_model_v4.pt` (epoch 132, obs mIoU=0.205)
- `checkpoints_v4/pointnet_lite_v4.onnx` (0.03 MB)
- `checkpoints_v4/training_curves_v4.png`

**Décision :** Verrouiller le modèle v4 et passer au pipeline d'inférence (Story 3). Itération training possible plus tard (plus d'epochs, focus electric_pole).

---

### 2026-02-19 — Story 3 : Pipeline d'inférence

#### Architecture du pipeline

```
HDF5 → get_frame_boundaries() → pour chaque frame :
  → read_frame_for_inference() → features (N, 5)
  → predict_frame() [chunked 65k pts] → predictions (N,)
  → predictions_to_boxes() [DBSCAN + PCA] → boxes
  → boxes_to_csv_lines() → append CSV
  → del + gc.collect()
```

#### Décisions techniques

| Décision | Choix | Raison |
|----------|-------|--------|
| Backend | PyTorch (pas ONNX) | Plus simple, pas de dep supplémentaire |
| Chunking | 65536 pts/pass, batch=1 | Tient sur T4 (15 GB), ~9 chunks/frame |
| Confidence | argmax direct (pas de seuil) | DBSCAN filtre déjà le bruit |
| Code | Tout inline dans chaque fichier | Évite les problèmes de path Colab |
| CSV format | Airbus deliverable format | ego_x/y/z/yaw + bbox center/dim/yaw + class_ID/label |

#### Test pipeline (modèle random, 50 frames sur scene_8)

- Pipeline validé end-to-end : HDF5 → segmentation → DBSCAN → CSV
- CSV format correct (header Airbus, labels capitalisés)
- ~60 boxes/frame avec modèle random (attendu ~5-8 avec vrai modèle)
- Pas d'OOM sur CPU (14 Go RAM)
- CPU : ~6s/frame (lecture 1s + inférence 4s + clustering 1s)
- GPU estimé : ~1-3s/frame

#### Fichiers produits

| Fichier | Description |
|---------|-------------|
| `scripts/inference.py` | Script standalone CLI (tout inline, pas de `src/`) |
| `notebooks/05_inference.ipynb` | Notebook Colab (même code + visu) |

**Utilisation :**
```bash
# Single scene:
python scripts/inference.py --input data/scene_8.h5 --checkpoint checkpoints_v4/best_model_v4.pt --output-dir outputs/pred_v4/
# All scenes:
python scripts/inference.py --input data/ --checkpoint checkpoints_v4/best_model_v4.pt --output-dir outputs/pred_v4/
```

**Format CSV de sortie (Airbus deliverable) :**
```
ego_x,ego_y,ego_z,ego_yaw,bbox_center_x,bbox_center_y,bbox_center_z,bbox_width,bbox_length,bbox_height,bbox_yaw,class_ID,class_label
```

---

### À documenter dans les prochaines étapes

- [x] Story 1.3/1.4 : FAIT
- [x] Story 2.1-2.3 v1→v4 : FAIT (obs mIoU: 0.05→0.03→0.168→0.205)
- [x] **Story 3 : Pipeline d'inférence** — FAIT (scripts/inference.py + notebooks/05_inference.ipynb)
- [ ] Story 2.4 : Courbe de robustesse densité (100%→25%)
- [ ] Itération training v5 (optionnel, si le temps le permet)
- [ ] D-7 : Résultats sur les fichiers d'évaluation
