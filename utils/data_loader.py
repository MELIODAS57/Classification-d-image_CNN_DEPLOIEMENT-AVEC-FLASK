"""
utils/data_loader.py
--------------------
Chargement et prétraitement du dataset CIFAR-10.

Fonctions exportées :
    load_and_preprocess()   → charge CIFAR-10 et normalise les pixels (0→1)
    build_tf_dataset()      → construit un tf.data.Dataset optimisé
    get_class_names()       → retourne la liste des 10 noms de classes
"""

import numpy as np
import tensorflow as tf


# ── Noms des 10 classes CIFAR-10 ────────────────────────────────────────────
CLASS_NAMES = [
    "avion",       # 0
    "voiture",     # 1
    "oiseau",      # 2
    "chat",        # 3
    "cerf",        # 4
    "chien",       # 5
    "grenouille",  # 6
    "cheval",      # 7
    "bateau",      # 8
    "camion",      # 9
]


def get_class_names():
    """Retourne la liste des noms de classes CIFAR-10."""
    return CLASS_NAMES


def load_and_preprocess():
    """
    Charge le dataset CIFAR-10 via tf.keras.datasets et normalise les pixels.

    La normalisation consiste à diviser les valeurs entières (0-255)
    par 255.0 pour obtenir des flottants dans [0.0, 1.0].

    Retours
    -------
    (x_train, y_train) : tuple
        x_train : np.ndarray de forme (50000, 32, 32, 3), dtype float32
        y_train : np.ndarray de forme (50000, 1), dtype uint8

    (x_test, y_test) : tuple
        x_test  : np.ndarray de forme (10000, 32, 32, 3), dtype float32
        y_test  : np.ndarray de forme (10000, 1), dtype uint8
    """
    print("[data_loader] Chargement de CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # ── Normalisation des pixels : 0-255 → 0.0-1.0 ──────────────────────────
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    print(f"[data_loader] Train : {x_train.shape}  |  Test : {x_test.shape}")
    print(f"[data_loader] Pixels — min : {x_train.min():.3f}  max : {x_train.max():.3f}")
    return (x_train, y_train), (x_test, y_test)


def build_tf_dataset(x, y, batch_size: int = 64, shuffle: bool = True):
    """
    Construit un pipeline tf.data.Dataset efficace.

    Utilise :
        - from_tensor_slices  : crée un dataset à partir de tableaux NumPy
        - shuffle             : mélange aléatoire (uniquement train)
        - batch               : regroupe en mini-lots
        - prefetch            : précharge le prochain lot en parallèle (AUTOTUNE)

    Paramètres
    ----------
    x          : np.ndarray — images normalisées
    y          : np.ndarray — labels entiers
    batch_size : int — taille des mini-lots (défaut 64)
    shuffle    : bool — mélanger les données (True pour train, False pour test)

    Retour
    ------
    tf.data.Dataset prêt à être passé à model.fit() ou model.evaluate()
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        # buffer_size = taille du dataset pour un shuffle parfait
        dataset = dataset.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)

    dataset = (
        dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)   # Optimisation automatique du prefetch
    )
    return dataset
