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

### 2026-02-20 — Post-processing + Training v5

#### Post-processing ajouté au pipeline d'inférence

Améliorations ajoutées à `scripts/inference.py` et `notebooks/05_inference.ipynb` :

| Étape | Paramètre | Description |
|-------|-----------|-------------|
| Confidence filter | seuil = 0.3 | Softmax sur les logits, points < 0.3 reclassés en background |
| Size filter | min_points par classe | Supprime les micro-clusters (bruit DBSCAN) |
| Size filter | max_dim par classe | Supprime les boxes aberrantes (over-merging) |
| NMS | IoU > 0.3 | Supprime les doublons, garde le cluster avec le plus de points |

**Seuils calibrés depuis les statistiques GT boxes :**

| Classe | min_points | max_dim (m) |
|--------|-----------|-------------|
| antenna | 5 | 80 |
| cable | 3 | 500 |
| electric_pole | 5 | 60 |
| wind_turbine | 10 | 200 |

#### Training v5 — Resume from v4

**Changements v4 → v5 :**

| Aspect | v4 | v5 |
|--------|----|----|
| Epochs | 150 (from scratch) | **+150** (resume from v4 best) |
| LR | 3e-4 | **2e-4** (plus conservateur) |
| Drop augment | 0-30% | **0-50%** (+ agressif pour robustesse) |
| Cycle restart | 50 | **75** |
| GPU | A100 80GB | A100 80GB |
| Temps | 131.6 min | **130.7 min** |

**Résultats v5 (best epoch 253, soit epoch 103 du fine-tuning) :**

| Classe | v1 | v3 | v4 | **v5** | Tendance |
|--------|-----|------|--------|--------|----------|
| background | 0.855 | 0.787 | 0.768 | **0.765** | stable |
| antenna | 0.005 | 0.290 | 0.256 | **0.310** | +21% |
| cable | 0.005 | 0.283 | 0.378 | **0.397** | +5% |
| electric_pole | 0.005 | 0.041 | 0.017 | **0.004** | pire |
| wind_turbine | 0.199 | 0.059 | 0.170 | **0.136** | -20% |
| **obstacle mIoU** | 0.05 | 0.168 | 0.205 | **0.212** | **+3.3%** |

- **1,882,693 paramètres** (inchangé)
- 2 warm restarts effectifs (epoch 228 = epoch 78 du fine-tuning a déclenché le meilleur résultat)
- Courbe en plateau après epoch 253 — le modèle a convergé

**Analyse v5 :**
- **antenna** : meilleure performance jamais atteinte (0.310) — le drop augment 0-50% aide la généralisation
- **cable** : toujours la meilleure classe obstacle (0.397), continue de progresser
- **electric_pole** : pire résultat de toutes les versions (0.004) — cette classe est structurellement trop difficile (21 pts médians, absent de scene_5)
- **wind_turbine** : régression vs v4 (0.136 vs 0.170) — possible compromis antenna/turbine
- **Gain global modeste** (+3.3%) : le modèle approche sa capacité maximale avec cette architecture

**Progression complète :**
```
v1 (0.05) → v2 (0.03) → v3 (0.168) → v4 (0.205) → v5 (0.212)
```

**Fichiers produits :**
- `notebooks/04_training_v5.ipynb` — notebook standalone pour resume training
- `checkpoints_v5/best_model_v5.pt` — meilleur modèle (epoch 253, obs mIoU=0.212)

---

### 2026-02-20 — Post-processing v2 : Calibration + Bug fixes + TTA

#### Bug CRITIQUE corrigé : class_ID mapping

| Champ | Avant (INCORRECT) | Après (CORRECT) |
|-------|-------------------|-----------------|
| class_ID | 1, 2, 3, 4 | **0, 1, 2, 3** |
| class_label "Electric pole" | "Electric pole" | **"Electric Pole"** |
| class_label "Wind turbine" | "Wind turbine" | **"Wind Turbine"** |

**Impact** : 100% des détections auraient été mal classées à l'évaluation Airbus. Corrigé dans `scripts/inference.py`, `scripts/generate_visualizations.py`, `src/config.py`.

#### Améliorations post-processing (v6)

| Changement | Avant | Après | Impact |
|------------|-------|-------|--------|
| Box confidence filter | absent | **mean softmax ≥ 0.6** | Réduit faux positifs de ~80% |
| MIN_POINTS_PER_BOX | {5, 3, 4, 10} | **{15, 5, 10, 25}** | Élimine micro-clusters |
| DBSCAN min_samples | {10, 5, 10, 15} | **{15, 8, 12, 25}** | Clusters plus denses |
| TTA (optionnel) | absent | **4x rotation Z** | +robustesse (--tta flag) |

#### Calibration empirique sur scene_8 (validation)

| Seuil box_conf | Boxes/frame | GT (5.1/fr) | Observation |
|----------------|-------------|-------------|-------------|
| 0.5 | 27.8 | 5.1 | Trop de faux positifs |
| **0.6** | **5.2** | **5.1** | **Meilleur match (choisi)** |
| 0.7 | 1.1 | 5.1 | Trop agressif |

**Résultats scene_8 avec box_conf=0.6 :**

| Classe | GT | Pred | Observation |
|--------|-----|------|-------------|
| antenna | 111 | 481 | 4.3x — sur-prédit (modèle confond d'autres classes avec antenna) |
| cable | 285 | 28 | 0.1x — sous-détecté |
| electric_pole | 40 | 0 | Non détecté (IoU modèle = 0.004) |
| wind_turbine | 70 | 7 | 0.1x — sous-détecté |

**Analyse** : Le nombre total de boxes est bien calibré (5.2 vs 5.1/frame GT). Mais la distribution par classe reflète les limites du modèle (mIoU=0.212) — le modèle sur-prédit antenna au détriment des 3 autres classes. L'amélioration la plus importante viendrait d'un meilleur modèle, pas du post-processing.

#### Nouveaux fichiers / scripts

| Fichier | Description |
|---------|-------------|
| `scripts/test_density_robustness.py` | Test robustesse densité (100%, 75%, 50%, 25%) |

#### CLI inference.py — nouveaux flags

```bash
# Avec box confidence filtering (défaut 0.6) :
python scripts/inference.py --input data/ --checkpoint checkpoints_v5/best_model_v5.pt --output-dir outputs/pred/

# Avec TTA (4x plus lent mais plus robuste) :
python scripts/inference.py --input data/ --checkpoint checkpoints_v5/best_model_v5.pt --output-dir outputs/pred/ --tta

# Custom thresholds :
python scripts/inference.py --input data/ --checkpoint checkpoints_v5/best_model_v5.pt --output-dir outputs/pred/ --box-conf-threshold 0.5 --conf-threshold 0.3
```

#### Test de robustesse densité (critère #3 Airbus)

Sous-échantillonnage aléatoire des points à 75%, 50%, 25% — inference sur scene_8 (100 frames).

| Densité | Boxes | Boxes/frame | Rétention vs 100% | Frames avec détections |
|---------|-------|-------------|--------------------|-----------------------|
| **100%** | 508 | 5.1 | 100% | 63 |
| **75%** | 542 | 5.4 | 107% | 67 |
| **50%** | 477 | 4.8 | **94%** | 63 |
| **25%** | 339 | 3.4 | **67%** | 57 |

Distribution par classe à chaque densité :

| Densité | Antenna | Cable | Electric Pole | Wind Turbine |
|---------|---------|-------|---------------|--------------|
| 100% | 471 | 30 | 0 | 7 |
| 75% | 497 | 37 | 0 | 8 |
| 50% | 447 | 25 | 0 | 5 |
| 25% | 304 | 31 | 0 | 4 |

**Analyse :**
- **75% → stable** voire légèrement meilleur (le sous-échantillonnage réduit le bruit pour DBSCAN)
- **50% → -6%** — bonne robustesse, les détections principales sont préservées
- **25% → -33%** — dégradation significative mais encore 3.4 boxes/frame
- **electric_pole = 0** à toutes les densités — le modèle ne sait pas le détecter (IoU=0.004)
- **antenna domine** (~90% des détections) — le modèle confond les autres classes avec antenna

**Fichiers produits :**
- `scripts/test_density_robustness.py` — script de test
- `outputs/density_test/density_report_scene_8.txt` — rapport complet

#### Visualisations v6 (livrable #4)

10 PNGs régénérées avec le post-processing v6 (box confidence 0.6, DBSCAN resserré).
Beaucoup plus propres que les v5 (5 boxes/frame vs 244 avant).

**Fichiers :** `outputs/visualizations_v6/scene_8_frame*.png`

#### Notebook Colab D-day

`notebooks/05_inference.ipynb` entièrement réécrit pour le jour J :
- Tout inline (pas de dépendance `src/`)
- GPU auto-détecté
- TTA optionnel (`USE_TTA = True/False`)
- Validation CSV automatique (class_ID 0-3, labels corrects)
- Checklist jour J intégrée
- Un seul changement à faire : `INPUT_DIR` → `eval_data/`, `SINGLE_SCENE = None`

---

### 2026-02-20 — Post-processing v7.3 : Per-class threshold calibration

#### Problème identifié

Le post-processing v6 avec un seuil global `BOX_CONFIDENCE_THRESHOLD = 0.6` produisait un total de boxes correct (~5.2/frame) mais avec une distribution par classe déséquilibrée :
- **antenna sur-prédite** : 481 vs 111 GT (4.3x)
- **cable sous-détecté** : 28 vs 285 GT (0.1x)
- **electric_pole non détecté** : 0 vs 40 GT
- **wind_turbine sous-détecté** : 7 vs 70 GT (0.1x)

Le modèle classifie antenna par défaut, et le seuil global élimine les boxes cable/turbine/pole (peu de points → confiance faible).

#### Approche : seuils per-class + reclassification géométrique

**3 modifications sans re-training :**

1. **Seuils de confiance per-class** (point-level et box-level) au lieu de global
2. **Reclassification géométrique** : antenna allongée+plate → cable, antenna grande+dense → wind_turbine
3. **DBSCAN/MIN_POINTS baissés** pour cable/pole/turbine (clusters plus petits que antenna)

#### Itérations de calibration sur scene_8 (100 frames, GT = 506 boxes)

| Version | antenna | cable | pole | turbine | **Total** | Ratio GT |
|---------|---------|-------|------|---------|-----------|----------|
| **v6 (avant)** | 481 | 28 | 0 | 7 | 516 | 1.02x |
| v7 (1er essai) | 81 | 2807 | 1277 | 349 | 4514 | 8.9x |
| v7.1 | 87 | 879 | 49 | 191 | 1206 | 2.4x |
| v7.2 | 85 | 49 | 51 | 57 | 242 | 0.48x |
| **v7.3 (final)** | **81** | **295** | **53** | **56** | **485** | **0.96x** |
| **GT** | 111 | 285 | 40 | 70 | 506 | — |

#### Seuils finaux v7.3

**Per-point confidence threshold** (softmax min pour garder un point) :

| Classe | Seuil |
|--------|-------|
| antenna | 0.40 |
| cable | 0.27 |
| electric_pole | 0.25 |
| wind_turbine | 0.30 |

**Per-box confidence threshold** (mean softmax du cluster) :

| Classe | Seuil |
|--------|-------|
| antenna | 0.70 |
| cable | 0.55 |
| electric_pole | 0.45 |
| wind_turbine | 0.60 |

**DBSCAN min_samples** : antenna=15, cable=5, pole=8, turbine=20
**MIN_POINTS_PER_BOX** : antenna=15, cable=3, pole=5, turbine=15

#### Reclassification géométrique

Fonction `reclassify_by_geometry()` appliquée après clustering, avant le filtre confidence :
- Si box antenna avec **ratio longueur/largeur > 5** et **hauteur < 1m** → reclassée en **cable**
- Si box antenna avec **dimension > 15m** et **> 200 points** → reclassée en **wind_turbine**

#### Résultats v7.3 vs v6

| Métrique | v6 | v7.3 | Amélioration |
|----------|-----|------|-------------|
| antenna | 481 (4.3x GT) | 81 (0.73x GT) | distribution réaliste |
| cable | 28 (0.1x GT) | 295 (1.04x GT) | **x10 détections** |
| electric_pole | 0 | 53 (1.33x GT) | **détecté pour la 1ère fois** |
| wind_turbine | 7 (0.1x GT) | 56 (0.80x GT) | **x8 détections** |
| Total | 516 | 485 | ratio 0.96x vs GT |
| Temps (scene_8) | ~80s | 73s | comparable |

**Impact majeur** : electric_pole passe de 0 à 53 détections — la classe la plus faible du modèle (IoU=0.004) est maintenant détectée grâce aux seuils permissifs et à la reclassification.

#### Robustesse densité v7.3 (scene_8, 100 frames)

| Densité | Boxes | Boxes/fr | Rétention | Antenna | Cable | Pole | Turbine |
|---------|-------|----------|-----------|---------|-------|------|---------|
| 100% | 487 | 4.9 | 100% | 91 | 288 | 51 | 57 |
| 75% | 459 | 4.6 | 94.3% | 90 | 274 | 43 | 52 |
| 50% | 406 | 4.1 | 83.4% | 82 | 246 | 43 | 35 |
| 25% | 387 | 3.9 | **79.5%** | 53 | 254 | 50 | 30 |

**Amélioration vs v6** : rétention à 25% passe de 67% à **79.5%** (+12 points). Toutes les classes détectées à toutes les densités (vs 0 electric_pole en v6). Cable particulièrement robuste (88.2% à 25%).

#### Fichiers modifiés

| Fichier | Modifications |
|---------|--------------|
| `scripts/inference.py` | Seuils per-class, reclassification géométrique, DBSCAN params |
| `scripts/generate_visualizations.py` | Mêmes seuils, même reclassification |
| `scripts/test_density_robustness.py` | Mêmes seuils, même reclassification |
| `notebooks/05_inference.ipynb` | Notebook Colab D-day avec seuils v7.3 |
| `notebooks/06_density_robustness.ipynb` | Notebook Colab test densité |
| `notebooks/07_visualizations.ipynb` | Notebook Colab visualisations |

---

### À documenter dans les prochaines étapes

- [x] Story 1.3/1.4 : FAIT
- [x] Story 2.1-2.3 v1→v5 : FAIT (obs mIoU: 0.05→0.03→0.168→0.205→0.212)
- [x] **Story 3 : Pipeline d'inférence + post-processing** — FAIT
- [x] **Post-processing v2 (v6)** : bug class_ID, box confidence, DBSCAN tuning, TTA — FAIT
- [x] **Post-processing v3 (v7.3)** : per-class thresholds, geometric reclassification — FAIT
- [x] **Story 2.4 : Robustesse densité** — v7.3 FAIT (83.4% à 50%, **79.5% à 25%**)
- [x] **Notebook Colab D-day** — FAIT (v7.3)
- [x] **Visualisations v6** — FAIT (10 PNGs) — à regénérer avec v7.3
- [ ] Regénérer visualisations avec v7.3
- [ ] Relancer test robustesse densité avec v7.3
- [ ] D-7 : Résultats sur les fichiers d'évaluation
