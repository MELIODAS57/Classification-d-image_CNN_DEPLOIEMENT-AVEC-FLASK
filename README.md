# Projet Fil Rouge — Mission 1 : Classification d'Images CNN

**Framework** : TensorFlow / Keras  
**Dataset** : CIFAR-10 (10 classes, 60 000 images 32×32)  
**Objectif** : Accuracy ≥ 70% sur le jeu de test

---

## Structure du projet (Skeleton)

```
projet_cnn/
├── data/               # Dossier des données brutes (CIFAR-10 auto-téléchargé)
├── models/
│   ├── __init__.py
│   └── cnn_model.py    # Classe CustomCNN (API Subclassing tf.keras.Model)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py  # Chargement CIFAR-10 + pipeline tf.data.Dataset
│   └── visualize.py    # Courbes d'apprentissage + matrice de confusion
├── train.py            # Script principal : compilation + entraînement + callbacks
├── evaluate.py         # Chargement modèle .keras + évaluation + figures
├── requirements.txt    # Dépendances Python
└── README.md
```

---

## Architecture du modèle

| Couche              | Paramètre clé              | Sortie (H×W×C)   |
|---------------------|----------------------------|-----------------|
| Input               | —                          | 32 × 32 × 3     |
| Data Augmentation   | RandomFlip, Rotation, Zoom | 32 × 32 × 3     |
| Conv2D + BN + ReLU  | 32 filtres, 3×3, same      | 32 × 32 × 32    |
| MaxPooling2D        | 2×2                        | 16 × 16 × 32    |
| Conv2D + BN + ReLU  | 64 filtres, 3×3, same      | 16 × 16 × 64    |
| MaxPooling2D        | 2×2                        | 8 × 8 × 64      |
| Conv2D + BN + ReLU  | 128 filtres, 3×3, same     | 8 × 8 × 128     |
| MaxPooling2D        | 2×2                        | 4 × 4 × 128     |
| Flatten             | —                          | 2 048           |
| Dense + ReLU        | 256 unités                 | 256             |
| Dropout             | taux = 0.5                 | 256             |
| Dense + Softmax     | 10 classes                 | 10              |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Entraînement
```bash
python train.py
```
Lance l'entraînement avec EarlyStopping (patience=12) et sauvegarde le meilleur modèle dans `saved_model/best_model.keras`.

### Évaluation
```bash
python evaluate.py
```
Charge le modèle sauvegardé, calcule la précision sur le test, génère la matrice de confusion et les courbes d'apprentissage dans `figures/`.

---

## Résultats attendus

- **Accuracy test** : ≥ 70%
- **Figures générées** :
  - `figures/courbes_apprentissage.png`
  - `figures/confusion_matrix.png`
  - `figures/sample_predictions.png`
