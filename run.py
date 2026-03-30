"""
run.py
------
Point d'entrée pour lancer l'application web Flask.

Usage :
    python run.py

L'application sera accessible sur : http://127.0.0.1:5000
"""

import sys
import os

# S'assurer que le dossier racine est dans le path Python
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CIFAR-10 CNN Classifier — Serveur Flask")
    print("="*55)
    print("  Ouvrez votre navigateur sur :")
    print("  → http://127.0.0.1:5000")
    print("\n  Appuyez sur Ctrl+C pour arrêter le serveur.")
    print("="*55 + "\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,    # False en déploiement local
    )
