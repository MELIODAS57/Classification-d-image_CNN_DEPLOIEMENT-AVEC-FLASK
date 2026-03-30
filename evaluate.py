"""
evaluate.py
-----------
Script d'évaluation du modèle CNN sauvegardé sur le jeu de test CIFAR-10.

Usage :
    python evaluate.py

Ce script :
    1. Charge le modèle sauvegardé (.keras)
    2. Charge les données de test CIFAR-10
    3. Calcule et affiche la précision sur le test
    4. Génère la matrice de confusion
    5. Affiche le rapport de classification détaillé (precision/recall/F1)
    6. Affiche des exemples de prédictions
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from models.cnn_model  import CustomCNN
from utils.data_loader import load_and_preprocess, CLASS_NAMES
from utils.visualize   import (
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_history,
)
import pickle, os

# ── Chemin du modèle ─────────────────────────────────────────────────────────
MODEL_PATH   = "saved_model/best_model.keras"
HISTORY_PATH = "saved_model/history.pkl"

# ── 1. Chargement du modèle sauvegardé ──────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}\n"
        "Lancez d'abord : python train.py"
    )

print(f"[evaluate] Chargement du modèle : {MODEL_PATH}")

# IMPORTANT — API Subclassing : il faut passer un batch factice AVANT
# d'appeler evaluate() ou predict(), sinon le graphe n'est pas construit
# et le modèle retourne des valeurs aléatoires (accuracy ~10%).
model = CustomCNN(num_classes=10)
model(tf.zeros((1, 32, 32, 3)), training=False)   # ← reconstruit le graphe
model.load_weights(MODEL_PATH)                     # ← charge les vrais poids

# Le modèle doit être compilé avant evaluate()/predict() sur un objet Model subclass
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Afficher le résumé avec les dimensions réelles
print()
model.build_graph(input_shape=(32, 32, 3)).summary()
print()

# ── 2. Chargement des données de test ────────────────────────────────────────
(_, _), (x_test, y_test) = load_and_preprocess()
y_true = y_test.flatten()   # (10000,) — vecteur 1D

# ── 3. Évaluation globale ────────────────────────────────────────────────────
print("\n[evaluate] Évaluation sur le jeu de test...")
loss, accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=0)

print("\n" + "="*60)
print(f"  Perte sur le test     : {loss:.4f}")
print(f"  Précision sur le test : {accuracy*100:.2f}%")
seuil = "✅ Objectif 70% atteint !" if accuracy >= 0.70 else "❌ Objectif 70% non atteint."
print(f"  {seuil}")
print("="*60 + "\n")

# ── 4. Prédictions ───────────────────────────────────────────────────────────
print("[evaluate] Génération des prédictions...")
y_pred_proba = model.predict(x_test, batch_size=64, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)   # (10000,) — classe prédite

# ── 5. Rapport de classification ─────────────────────────────────────────────
print("\n[evaluate] Rapport de classification détaillé :")
print("-"*60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ── 6. Matrice de confusion ───────────────────────────────────────────────────
print("[evaluate] Génération de la matrice de confusion...")
plot_confusion_matrix(y_true, y_pred, save=True)

# ── 7. Exemples de prédictions ────────────────────────────────────────────────
print("[evaluate] Affichage d'exemples de prédictions...")
plot_sample_predictions(x_test, y_test, y_pred_proba, n=25, save=True)

# ── 8. Courbes d'apprentissage (depuis history sauvegardé) ───────────────────
if os.path.exists(HISTORY_PATH):
    print("[evaluate] Chargement et affichage des courbes d'apprentissage...")
    with open(HISTORY_PATH, "rb") as f:
        history_dict = pickle.load(f)
    plot_history(history_dict, save=True)

print("\n[evaluate] Évaluation terminée. Figures disponibles dans : figures/")
