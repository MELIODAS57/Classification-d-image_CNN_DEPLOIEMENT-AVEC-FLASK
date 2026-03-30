"""
explore_data.py
---------------
Script d'exploration et de visualisation du dataset CIFAR-10.
À lancer AVANT train.py pour comprendre les données.

Usage :
    python explore_data.py

Figures générées dans figures/ :
    1. dataset_samples.png      → grille d'images organisée par classe
    2. class_distribution.png   → répartition des 10 classes (barres + camembert)
    3. pixel_statistics.png     → distribution pixels avant/après normalisation
    4. augmentation_preview.png → aperçu de la Data Augmentation
    5. mean_images.png          → image moyenne par classe
"""

import numpy as np
import tensorflow as tf

from utils.data_loader import CLASS_NAMES
from utils.visualize import (
    plot_dataset_samples,
    plot_class_distribution,
    plot_pixel_statistics,
    plot_augmentation_preview,
    plot_mean_images,
)

print("=" * 60)
print("  EXPLORATION DU DATASET CIFAR-10")
print("=" * 60)

# ── 1. Chargement des données BRUTES (avant normalisation) ───────────────────
print("\n[explore] Chargement de CIFAR-10...")
(x_train_raw, y_train), (x_test_raw, y_test) = tf.keras.datasets.cifar10.load_data()

# ── 2. Informations générales ────────────────────────────────────────────────
print("\n[explore] Informations sur le dataset :")
print(f"  x_train shape  : {x_train_raw.shape}   ← (50000 images, 32px, 32px, 3 canaux RGB)")
print(f"  y_train shape  : {y_train.shape}        ← (50000 labels entiers, 1 colonne)")
print(f"  x_test  shape  : {x_test_raw.shape}    ← (10000 images de test)")
print(f"  y_test  shape  : {y_test.shape}         ← (10000 labels de test)")
print(f"  Dtype images   : {x_train_raw.dtype}              ← entiers 0-255 avant normalisation")
print(f"  Dtype labels   : {y_train.dtype}")
print(f"  Valeur min px  : {x_train_raw.min()}")
print(f"  Valeur max px  : {x_train_raw.max()}")

print("\n[explore] Répartition par classe (train) :")
y_flat = y_train.flatten()
for i, name in enumerate(CLASS_NAMES):
    count = np.sum(y_flat == i)
    bar = "█" * (count // 200)
    print(f"  [{i}] {name:<12} : {count:5d} images  {bar}")

# ── 3. Normalisation (pour les visualisations qui en ont besoin) ─────────────
x_train_norm = x_train_raw.astype("float32") / 255.0
x_test_norm  = x_test_raw.astype("float32")  / 255.0

print("\n[explore] Après normalisation :")
print(f"  Valeur min px  : {x_train_norm.min():.3f}")
print(f"  Valeur max px  : {x_train_norm.max():.3f}")
print(f"  Dtype images   : {x_train_norm.dtype}")

# ── 4. Génération des figures ─────────────────────────────────────────────────
print("\n[explore] Génération des figures...")

print("\n  1/5 — Grille d'exemples par classe...")
plot_dataset_samples(x_train_norm, y_train, n_per_class=8, save=True)

print("\n  2/5 — Distribution des classes...")
plot_class_distribution(y_train, y_test, save=True)

print("\n  3/5 — Statistiques des pixels (avant/après normalisation)...")
plot_pixel_statistics(x_train_raw, x_train_norm, save=True)

print("\n  4/5 — Aperçu de la Data Augmentation...")
plot_augmentation_preview(x_train_norm, y_train, n_augmentations=6, save=True)

print("\n  5/5 — Images moyennes par classe...")
plot_mean_images(x_train_norm, y_train, save=True)

# ── 5. Résumé final ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RÉSUMÉ — Ce que vous devez retenir de l'exploration")
print("=" * 60)
print("""
  Dataset     : CIFAR-10
  Train       : 50 000 images (5 000 par classe, dataset parfaitement équilibré)
  Test        : 10 000 images (1 000 par classe)
  Résolution  : 32 x 32 pixels x 3 canaux RGB → 3 072 valeurs par image
  Labels      : entiers de 0 à 9 (SparseCategoricalCrossentropy les accepte directement)
  Avant norm. : pixels entre 0 et 255 (uint8)
  Après norm. : pixels entre 0.0 et 1.0 (float32) ← ce qu'on passe au CNN

  Figures sauvegardées dans : figures/
    ✅ dataset_samples.png
    ✅ class_distribution.png
    ✅ pixel_statistics.png
    ✅ augmentation_preview.png
    ✅ mean_images.png

  → Lancez maintenant : python train.py
""")
