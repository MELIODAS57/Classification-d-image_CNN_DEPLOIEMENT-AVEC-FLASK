"""
app/routes.py
-------------
Toutes les routes Flask de l'application de déploiement CNN.

Routes :
    GET  /                → Page principale (interface utilisateur)
    POST /predict/upload  → Prédiction par upload de fichier
    POST /predict/webcam  → Prédiction par capture webcam (base64)
    POST /predict/url     → Prédiction par URL d'image
    GET  /health          → Vérification que le serveur est actif
"""

import base64
import io
import traceback

from flask import Blueprint, jsonify, render_template, request
from PIL import Image

from .predict import predict_from_bytes, predict_from_url

bp = Blueprint("main", __name__)


# ── Page principale ──────────────────────────────────────────────────────────
@bp.route("/")
def index():
    return render_template("index.html")


# ── Route 1 : Upload de fichier ───────────────────────────────────────────────
@bp.route("/predict/upload", methods=["POST"])
def predict_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier reçu."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Fichier vide."}), 400

        allowed = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in allowed:
            return jsonify({"error": f"Format non supporté : .{ext}"}), 400

        image_bytes = file.read()
        result = predict_from_bytes(image_bytes)

        # Retourner aussi l'image en base64 pour l'afficher dans l'UI
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        result["image_b64"] = f"data:image/{ext};base64,{img_b64}"

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Route 2 : Webcam (image base64 depuis le navigateur) ─────────────────────
@bp.route("/predict/webcam", methods=["POST"])
def predict_webcam():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Aucune image reçue."}), 400

        # Format attendu : "data:image/jpeg;base64,/9j/4AAQ..."
        header, encoded = data["image"].split(",", 1)
        image_bytes = base64.b64decode(encoded)

        result = predict_from_bytes(image_bytes)
        result["image_b64"] = data["image"]   # renvoyer pour affichage

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Route 3 : URL d'image ─────────────────────────────────────────────────────
@bp.route("/predict/url", methods=["POST"])
def predict_url():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Aucune URL reçue."}), 400

        url = data["url"].strip()
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": "URL invalide (doit commencer par http:// ou https://)"}), 400

        result = predict_from_url(url)
        result["image_b64"] = url   # on passe l'URL directement à <img src>

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Route santé ───────────────────────────────────────────────────────────────
@bp.route("/health")
def health():
    return jsonify({"status": "ok", "model": "CustomCNN CIFAR-10"})
