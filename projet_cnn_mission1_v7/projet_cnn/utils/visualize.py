"""
utils/visualize.py
------------------
Fonctions de visualisation pour le projet CNN CIFAR-10.

Fonctions exportées :
    --- Exploration des données ---
    plot_dataset_samples()    → grille d'images par classe (vue d'ensemble)
    plot_class_distribution() → histogramme de répartition des 10 classes
    plot_pixel_statistics()   → distribution des valeurs de pixels (avant/après normalisation)
    plot_augmentation_preview()→ aperçu des transformations de Data Augmentation
    plot_mean_images()        → image moyenne par classe

    --- Résultats d'entraînement ---
    plot_history()            → courbes Train/Val Loss et Accuracy
    plot_confusion_matrix()   → matrice de confusion annotée (seaborn)
    plot_sample_predictions() → grille d'images avec prédictions vs vérité
"""

import os
import matplotlib
matplotlib.use("Agg")   # ← Backend sans fenêtre (fonctionne partout : Windows, Linux, serveur)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from utils.data_loader import CLASS_NAMES


# ── Répertoire de sortie des figures ────────────────────────────────────────
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette de couleurs cohérente pour les 10 classes ───────────────────────
CLASS_COLORS = [
    "#185FA5", "#0F6E56", "#EF9F27", "#A32D2D", "#534AB7",
    "#D85A30", "#1D9E75", "#BA7517", "#E24B4A", "#3B6D11",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VISUALISATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

def plot_dataset_samples(x, y, n_per_class: int = 8, save: bool = True) -> None:
    """
    Affiche une grille d'images organisée par classe.
    Chaque ligne = une classe CIFAR-10, chaque colonne = un exemple.

    Paramètres
    ----------
    x            : np.ndarray — images normalisées (0-1), shape (N, 32, 32, 3)
    y            : np.ndarray — labels entiers, shape (N, 1) ou (N,)
    n_per_class  : int — nombre d'exemples par classe à afficher (défaut 8)
    save         : bool — sauvegarde la figure si True
    """
    y_flat = y.flatten()
    n_classes = len(CLASS_NAMES)

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 1.6, n_classes * 1.8)
    )
    fig.suptitle(
        f"CIFAR-10 — {n_per_class} exemples par classe\n"
        f"({len(x):,} images au total, 32×32 pixels RGB)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Indices de toutes les images appartenant à cette classe
        indices = np.where(y_flat == cls_idx)[0]
        # Choisir n_per_class exemples aléatoires
        chosen = np.random.choice(indices, size=n_per_class, replace=False)

        for col, img_idx in enumerate(chosen):
            ax = axes[cls_idx, col]
            ax.imshow(x[img_idx])
            ax.axis("off")

            # Étiquette de classe sur la première colonne uniquement
            if col == 0:
                ax.set_ylabel(
                    f"{cls_idx} — {cls_name}",
                    fontsize=9, fontweight="bold",
                    color=CLASS_COLORS[cls_idx],
                    rotation=0, labelpad=60, va="center"
                )

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "dataset_samples.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Grille d'exemples sauvegardée → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_class_distribution(y_train, y_test, save: bool = True) -> None:
    """
    Affiche la répartition des images par classe (train + test).
    Vérifie visuellement que le dataset est bien équilibré.

    Paramètres
    ----------
    y_train : np.ndarray — labels d'entraînement
    y_test  : np.ndarray — labels de test
    save    : bool — sauvegarde si True
    """
    y_train_flat = y_train.flatten()
    y_test_flat  = y_test.flatten()

    counts_train = [np.sum(y_train_flat == i) for i in range(len(CLASS_NAMES))]
    counts_test  = [np.sum(y_test_flat  == i) for i in range(len(CLASS_NAMES))]

    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Répartition des classes — CIFAR-10", fontsize=13, fontweight="bold")

    # ── Graphique en barres côte à côte ──────────────────────────────────────
    ax = axes[0]
    bars_train = ax.bar(x_pos - width/2, counts_train, width,
                        color=CLASS_COLORS, alpha=0.85, label="Train (50 000)")
    bars_test  = ax.bar(x_pos + width/2, counts_test,  width,
                        color=CLASS_COLORS, alpha=0.45, label="Test (10 000)")

    # Annotations sur les barres
    for bar in bars_train:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{int(bar.get_height()):,}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Nombre d'images par classe")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(counts_train) * 1.15)

    # ── Camembert (train uniquement) ─────────────────────────────────────────
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts_train,
        labels=CLASS_NAMES,
        colors=CLASS_COLORS,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(8.5)
    ax2.set_title("Proportions (jeu d'entraînement)")

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "class_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Distribution des classes sauvegardée → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_pixel_statistics(x_train_raw, x_train_norm, save: bool = True) -> None:
    """
    Compare la distribution des valeurs de pixels avant et après normalisation.
    Montre aussi la moyenne par canal RGB.

    Paramètres
    ----------
    x_train_raw  : np.ndarray — images brutes (valeurs 0-255, dtype uint8)
    x_train_norm : np.ndarray — images normalisées (valeurs 0-1, dtype float32)
    save         : bool — sauvegarde si True
    """
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Statistiques des pixels — CIFAR-10", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    canal_noms    = ["Rouge (R)", "Vert (G)", "Bleu (B)"]
    canal_couleurs = ["#E24B4A",  "#3B6D11",  "#185FA5"]

    # ── Ligne 1 : histogrammes AVANT normalisation ────────────────────────────
    for i, (nom, couleur) in enumerate(zip(canal_noms, canal_couleurs)):
        ax = fig.add_subplot(gs[0, i])
        valeurs = x_train_raw[:, :, :, i].flatten()
        ax.hist(valeurs, bins=64, color=couleur, alpha=0.75, edgecolor="none")
        ax.set_title(f"Avant norm. — {nom}", fontsize=10)
        ax.set_xlabel("Valeur du pixel (0–255)")
        ax.set_ylabel("Fréquence")
        ax.axvline(np.mean(valeurs), color="black", linestyle="--",
                   linewidth=1.2, label=f"Moy. = {np.mean(valeurs):.1f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # ── Ligne 2 : histogrammes APRÈS normalisation ────────────────────────────
    for i, (nom, couleur) in enumerate(zip(canal_noms, canal_couleurs)):
        ax = fig.add_subplot(gs[1, i])
        valeurs = x_train_norm[:, :, :, i].flatten()
        ax.hist(valeurs, bins=64, color=couleur, alpha=0.75, edgecolor="none")
        ax.set_title(f"Après norm. — {nom}", fontsize=10)
        ax.set_xlabel("Valeur du pixel (0.0–1.0)")
        ax.set_ylabel("Fréquence")
        ax.axvline(np.mean(valeurs), color="black", linestyle="--",
                   linewidth=1.2, label=f"Moy. = {np.mean(valeurs):.3f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    if save:
        path = os.path.join(FIGURES_DIR, "pixel_statistics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Statistiques pixels sauvegardées → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_augmentation_preview(x, y, n_augmentations: int = 6, save: bool = True) -> None:
    """
    Montre l'effet de la Data Augmentation sur des images réelles.
    Pour chaque image originale, affiche n_augmentations versions transformées.

    Paramètres
    ----------
    x                : np.ndarray — images normalisées (0-1)
    y                : np.ndarray — labels entiers
    n_augmentations  : int — nombre de versions augmentées à montrer (défaut 6)
    save             : bool — sauvegarde si True
    """
    # Pipeline d'augmentation (identique à celui du modèle)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Choisir une image de chaque classe pour l'exemple (5 classes affichées)
    y_flat = y.flatten()
    classes_a_montrer = [0, 2, 3, 5, 7]   # avion, oiseau, chat, chien, cheval
    n_lignes = len(classes_a_montrer)
    n_cols   = 1 + n_augmentations         # 1 originale + n versions augmentées

    fig, axes = plt.subplots(n_lignes, n_cols, figsize=(n_cols * 1.6, n_lignes * 1.9))
    fig.suptitle(
        "Data Augmentation — Effet sur les images CIFAR-10\n"
        "(colonne 1 = image originale | colonnes 2+ = versions augmentées)",
        fontsize=11, fontweight="bold", y=1.02
    )

    for row, cls_idx in enumerate(classes_a_montrer):
        # Choisir une image de cette classe
        indices   = np.where(y_flat == cls_idx)[0]
        img_idx   = np.random.choice(indices)
        img_orig  = x[img_idx]                  # (32, 32, 3)
        img_batch = img_orig[np.newaxis, ...]    # (1, 32, 32, 3)

        # Colonne 0 : image originale
        ax = axes[row, 0]
        ax.imshow(img_orig)
        ax.axis("off")
        ax.set_title("Original", fontsize=8, fontweight="bold")
        ax.set_ylabel(
            CLASS_NAMES[cls_idx], fontsize=9, fontweight="bold",
            color=CLASS_COLORS[cls_idx], rotation=0, labelpad=48, va="center"
        )

        # Colonnes 1+ : versions augmentées
        for col in range(1, n_cols):
            img_aug = augmentation(img_batch, training=True)[0].numpy()
            img_aug = np.clip(img_aug, 0, 1)

            ax = axes[row, col]
            ax.imshow(img_aug)
            ax.axis("off")
            ax.set_title(f"Aug. #{col}", fontsize=7.5, color="#534AB7")

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "augmentation_preview.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Aperçu augmentation sauvegardé → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_mean_images(x, y, save: bool = True) -> None:
    """
    Affiche l'image moyenne de chaque classe.
    Révèle la forme 'typique' que le modèle doit apprendre par classe.

    Paramètres
    ----------
    x    : np.ndarray — images normalisées (0-1)
    y    : np.ndarray — labels entiers
    save : bool — sauvegarde si True
    """
    y_flat = y.flatten()
    n_classes = len(CLASS_NAMES)

    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    fig.suptitle(
        "Image moyenne par classe — CIFAR-10\n"
        "(représentation du 'prototype' visuel appris par le CNN)",
        fontsize=12, fontweight="bold"
    )

    for cls_idx, ax in enumerate(axes.flat):
        if cls_idx >= n_classes:
            ax.axis("off")
            continue

        # Moyenne de toutes les images de cette classe
        indices   = np.where(y_flat == cls_idx)[0]
        mean_img  = np.mean(x[indices], axis=0)
        count     = len(indices)

        ax.imshow(mean_img)
        ax.axis("off")
        ax.set_title(
            f"{CLASS_NAMES[cls_idx]}\n({count:,} images)",
            fontsize=9.5, fontweight="bold",
            color=CLASS_COLORS[cls_idx]
        )

        # Bordure colorée autour de l'image
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(CLASS_COLORS[cls_idx])
            spine.set_linewidth(2.5)

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "mean_images.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Images moyennes sauvegardées → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_history(history_dict: dict, save: bool = True) -> None:
    """
    Trace les courbes de perte et de précision issues de model.fit().

    Paramètres
    ----------
    history_dict : dict
        Dictionnaire history.history contenant les clés :
        'loss', 'val_loss', 'accuracy', 'val_accuracy'
    save : bool
        Si True, sauvegarde la figure dans figures/courbes_apprentissage.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'Apprentissage — CNN CIFAR-10", fontsize=14, fontweight="bold")

    epochs = range(1, len(history_dict["loss"]) + 1)

    # ── Courbe de perte (Loss) ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history_dict["loss"],     label="Train Loss",      color="#185FA5", linewidth=2)
    ax.plot(epochs, history_dict["val_loss"], label="Validation Loss", color="#E85D24", linewidth=2, linestyle="--")

    # Repère de l'arrêt (EarlyStopping)
    best_epoch = int(np.argmin(history_dict["val_loss"])) + 1
    ax.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Meilleur epoch ({best_epoch})")

    ax.set_title("Perte (Loss)")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Valeur de la perte")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Courbe de précision (Accuracy) ──────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history_dict["accuracy"],     label="Train Accuracy",      color="#0F6E56", linewidth=2)
    ax.plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy", color="#EF9F27", linewidth=2, linestyle="--")

    best_acc = max(history_dict["val_accuracy"])
    ax.axhline(y=0.70, color="#A32D2D", linestyle=":", linewidth=1.5, label="Seuil 70%")
    ax.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Meilleur epoch ({best_epoch})")

    ax.set_title("Précision (Accuracy)")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Précision")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "courbes_apprentissage.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Courbes sauvegardées → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save: bool = True) -> None:
    """
    Génère et affiche la matrice de confusion avec seaborn.

    Paramètres
    ----------
    y_true : array-like — vraies étiquettes (entiers 0-9)
    y_pred : array-like — étiquettes prédites (entiers 0-9)
    save   : bool — sauvegarde la figure si True
    """
    cm = confusion_matrix(y_true, y_pred)

    # Matrice normalisée (%) pour la couleur, valeurs brutes pour les annotations
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm_norm,
        annot=cm,              # Annotations = valeurs brutes
        fmt="d",               # Format entier
        cmap="Blues",
        linewidths=0.4,
        linecolor="lightgray",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"label": "% de la classe réelle"},
    )

    ax.set_title("Matrice de Confusion — CNN CIFAR-10", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Vraie classe", fontsize=12)
    ax.set_xlabel("Classe prédite", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Matrice de confusion sauvegardée → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()


def plot_sample_predictions(x_test, y_true, y_pred_proba, n: int = 25, save: bool = True) -> None:
    """
    Affiche une grille de n images avec leur prédiction et la vraie classe.
    Vert = bonne prédiction | Rouge = erreur.

    Paramètres
    ----------
    x_test        : np.ndarray — images de test normalisées (0-1)
    y_true        : np.ndarray — vraies étiquettes (entiers)
    y_pred_proba  : np.ndarray — probabilités en sortie du modèle
    n             : int — nombre d'images à afficher (défaut 25)
    save          : bool — sauvegarde si True
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_true.flatten()

    ncols = 5
    nrows = n // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8))
    fig.suptitle("Exemples de prédictions", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue

        img   = x_test[i]
        true  = y_true[i]
        pred  = y_pred[i]
        conf  = y_pred_proba[i][pred] * 100

        ax.imshow(img)
        ax.axis("off")

        color = "#0F6E56" if pred == true else "#A32D2D"
        label = (
            f"Pred: {CLASS_NAMES[pred]} ({conf:.0f}%)\n"
            f"Vrai: {CLASS_NAMES[true]}"
        )
        ax.set_title(label, fontsize=8, color=color, pad=3)

    green_patch = mpatches.Patch(color="#0F6E56", label="Correcte")
    red_patch   = mpatches.Patch(color="#A32D2D", label="Erreur")
    fig.legend(handles=[green_patch, red_patch], loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "sample_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[visualize] Prédictions exemples sauvegardées → {path}")

    # plt.show()  # désactivé : backend Agg ne supporte pas les fenêtres
    plt.close()
