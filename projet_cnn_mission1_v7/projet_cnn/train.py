"""
train.py
--------
Script principal d'entraînement du CNN sur CIFAR-10.

Usage :
    python train.py

Ce script :
    1. Charge et normalise les données CIFAR-10
    2. Construit les pipelines tf.data.Dataset
    3. Instancie le modèle CustomCNN
    4. Compile avec Adam + SparseCategoricalCrossentropy
    5. Entraîne avec EarlyStopping et ModelCheckpoint
    6. Sauvegarde le modèle (.keras) et l'historique (.pkl)
    7. Affiche les courbes d'apprentissage
"""

import os
import pickle
import tensorflow as tf

from models.cnn_model  import CustomCNN
from utils.data_loader import load_and_preprocess, build_tf_dataset
from utils.visualize   import plot_history

# ── Reproductibilité ─────────────────────────────────────────────────────────
tf.random.set_seed(42)

# ── Hyper-paramètres ─────────────────────────────────────────────────────────
BATCH_SIZE    = 64
# EPOCHS = plafond de sécurité uniquement.
# En pratique, EarlyStopping(patience=12) arrête l'entraînement bien avant.
# Sur CIFAR-10, la convergence survient généralement entre 30 et 50 époques.
# On met 100 pour être sûr de ne jamais bloquer une convergence tardive.
EPOCHS        = 100
PATIENCE      = 12       # Patience EarlyStopping
LEARNING_RATE = 1e-3     # Taux d'apprentissage initial Adam
MODEL_PATH    = "saved_model/best_model.keras"
HISTORY_PATH  = "saved_model/history.pkl"

os.makedirs("saved_model", exist_ok=True)

# ── 1. Chargement et prétraitement des données ───────────────────────────────
(x_train, y_train), (x_test, y_test) = load_and_preprocess()

# ── 2. Pipelines tf.data.Dataset ─────────────────────────────────────────────
# CIFAR-10 fournit officiellement :
#   - 50 000 images pour l'entraînement  → train_ds  (toutes utilisées)
#   - 10 000 images pour le test/val     → val_ds    (utilisées comme validation)
# On n'effectue AUCUNE re-découpe des 50 000 images d'entraînement.
train_ds = build_tf_dataset(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_ds   = build_tf_dataset(x_test,  y_test,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\n[train] Images d'entraînement : {len(x_train)} (50 000 — CIFAR-10 complet)")
print(f"[train] Images de validation  : {len(x_test)}  (10 000 — jeu de test CIFAR-10)")
print(f"[train] Train batches         : {len(train_ds)}")
print(f"[train] Val batches           : {len(val_ds)}")

# ── 3. Instanciation du modèle ───────────────────────────────────────────────
model = CustomCNN(num_classes=10)

# Appel factice pour construire le graphe et afficher le résumé
_ = model(tf.zeros((1, 32, 32, 3)), training=False)
print("\n" + "="*60)
model.build_graph(input_shape=(32, 32, 3)).summary()
print("="*60 + "\n")

# ── 4. Compilation ───────────────────────────────────────────────────────────
# - optimizer  : Adam (taux adaptatif, momentum)
# - loss       : SparseCategoricalCrossentropy (labels entiers, pas one-hot)
# - metrics    : accuracy (proportion de bonnes prédictions)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 5. Callbacks ─────────────────────────────────────────────────────────────
# EarlyStopping : arrête l'entraînement si val_loss ne s'améliore pas
# pendant PATIENCE époques, puis restaure les meilleurs poids.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

# ModelCheckpoint : sauvegarde le modèle dès qu'une amélioration est détectée
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

# ReduceLROnPlateau : réduit le LR si la val_loss stagne
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1,
)

callbacks = [early_stopping, checkpoint, reduce_lr]

# ── 6. Entraînement ──────────────────────────────────────────────────────────
print("[train] Démarrage de l'entraînement...\n")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)

# ── 7. Sauvegarde de l'historique ────────────────────────────────────────────
with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history.history, f)
print(f"\n[train] Historique sauvegardé → {HISTORY_PATH}")

# ── 8. Résumé final ──────────────────────────────────────────────────────────
best_val_acc = max(history.history["val_accuracy"])
epochs_ran   = len(history.history["loss"])

print("\n" + "="*60)
print(f"  Époques effectuées     : {epochs_ran}")
print(f"  Meilleure val_accuracy : {best_val_acc*100:.2f}%")
print(f"  Modèle sauvegardé      : {MODEL_PATH}")
print("="*60)

# ── 9. Visualisation des courbes ─────────────────────────────────────────────
plot_history(history.history, save=True)

print("\n[train] Entraînement terminé. Lancez maintenant : python evaluate.py")
