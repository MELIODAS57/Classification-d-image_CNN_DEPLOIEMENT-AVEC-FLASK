"""
app/predict.py
--------------
Moteur de prédiction — charge le modèle une seule fois au démarrage
et expose une fonction predict_image() utilisée par les routes Flask.
"""

import io
import os
import sys
import urllib.request

import numpy as np
from PIL import Image
import tensorflow as tf

# Ajouter la racine du projet au path pour importer CustomCNN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.cnn_model import CustomCNN

# ── Noms des classes CIFAR-10 (FR + EN) ─────────────────────────────────────
CLASS_NAMES_FR = [
    "Avion", "Voiture", "Oiseau", "Chat", "Cerf",
    "Chien", "Grenouille", "Cheval", "Bateau", "Camion",
]
CLASS_NAMES_EN = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]
CLASS_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌",
                "🐶", "🐸", "🐴", "🚢", "🚚"]

# ── Chemin du modèle ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "saved_model", "best_model.keras"
)

# ── Chargement du modèle (une seule fois au démarrage de Flask) ──────────────
_model = None

def get_model():
    """Charge et met en cache le modèle Keras."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}\n"
                "Lancez d'abord : python train.py"
            )
        print(f"[predict] Chargement du modèle : {MODEL_PATH}")

        # IMPORTANT — API Subclassing : reconstruire le graphe
        # avant de charger les poids, sinon accuracy ~10%
        _model = CustomCNN(num_classes=10)
        _model(tf.zeros((1, 32, 32, 3)), training=False)  # ← construit le graphe
        _model.load_weights(MODEL_PATH)                    # ← charge les poids

        print("[predict] Modèle chargé avec succès !")
    return _model


def _preprocess(pil_image: Image.Image) -> np.ndarray:
    """
    Prétraite une image PIL pour le CNN :
      1. Redimensionne en 32×32 pixels
      2. Convertit en RGB (ignore canal alpha si PNG)
      3. Normalise les pixels entre 0.0 et 1.0
      4. Ajoute la dimension batch → (1, 32, 32, 3)
    """
    img = pil_image.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0   # (32, 32, 3)
    return arr[np.newaxis, ...]                      # (1, 32, 32, 3)


def predict_image(pil_image: Image.Image) -> dict:
    """
    Prédit la classe d'une image PIL.

    Retour
    ------
    dict avec les clés :
        predicted_class  : str  — nom de la classe prédite (FR)
        predicted_en     : str  — nom anglais
        emoji            : str  — emoji de la classe
        confidence       : float — probabilité en % (ex: 87.3)
        all_probs        : list[dict] — probabilités pour les 10 classes,
                           triées par ordre décroissant
    """
    model  = get_model()
    tensor = _preprocess(pil_image)

    probs  = model.predict(tensor, verbose=0)[0]   # (10,)
    top_idx = int(np.argmax(probs))

    all_probs = sorted(
        [
            {
                "index":      i,
                "name_fr":    CLASS_NAMES_FR[i],
                "name_en":    CLASS_NAMES_EN[i],
                "emoji":      CLASS_EMOJIS[i],
                "probability": round(float(probs[i]) * 100, 2),
            }
            for i in range(10)
        ],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return {
        "predicted_class": CLASS_NAMES_FR[top_idx],
        "predicted_en":    CLASS_NAMES_EN[top_idx],
        "emoji":           CLASS_EMOJIS[top_idx],
        "confidence":      round(float(probs[top_idx]) * 100, 2),
        "all_probs":       all_probs,
    }


def predict_from_bytes(image_bytes: bytes) -> dict:
    """Prédit depuis des bytes bruts (upload ou webcam)."""
    pil_image = Image.open(io.BytesIO(image_bytes))
    return predict_image(pil_image)


def predict_from_url(url: str) -> dict:
    """Prédit depuis une URL d'image distante."""
    headers = {"User-Agent": "Mozilla/5.0"}
    req     = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        image_bytes = response.read()
    return predict_from_bytes(image_bytes)
