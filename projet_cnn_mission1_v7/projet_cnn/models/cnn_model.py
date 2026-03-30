"""
models/cnn_model.py
-------------------
Définition du modèle CustomCNN pour la classification CIFAR-10.
Utilise l'API Subclassing de Keras (tf.keras.Model).

Architecture :
    Data Augmentation
    → Bloc Conv 1 : Conv2D(32, 3x3, same) → BatchNorm → ReLU → MaxPool(2x2)
    → Bloc Conv 2 : Conv2D(64, 3x3, same) → BatchNorm → ReLU → MaxPool(2x2)
    → Bloc Conv 3 : Conv2D(128, 3x3, same) → BatchNorm → ReLU → MaxPool(2x2)
    → Flatten
    → Dense(256, relu) → Dropout(0.5)
    → Dense(10, softmax)
"""

import tensorflow as tf


class CustomCNN(tf.keras.Model):
    """
    Réseau CNN personnalisé pour la classification d'images CIFAR-10.

    Paramètres
    ----------
    num_classes : int
        Nombre de classes en sortie (10 pour CIFAR-10).
    """

    def __init__(self, num_classes: int = 10):
        super(CustomCNN, self).__init__()

        # ── Bloc Data Augmentation ──────────────────────────────────────────
        # Couches d'augmentation natives TensorFlow.
        # Elles sont automatiquement désactivées en mode inférence.
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ],
            name="data_augmentation",
        )

        # ── Bloc Conv 1 ─────────────────────────────────────────────────────
        # 32 filtres 3×3, padding='same' → conserve la taille spatiale (32×32)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,       # BatchNorm gère le biais
            name="conv1",
        )
        self.bn1   = tf.keras.layers.BatchNormalization(name="bn1")
        self.act1  = tf.keras.layers.Activation("relu", name="relu1")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")

        # ── Bloc Conv 2 ─────────────────────────────────────────────────────
        # 64 filtres 3×3, padding='same'
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            name="conv2",
        )
        self.bn2   = tf.keras.layers.BatchNormalization(name="bn2")
        self.act2  = tf.keras.layers.Activation("relu", name="relu2")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")

        # ── Bloc Conv 3 ─────────────────────────────────────────────────────
        # 128 filtres 3×3, padding='same'
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            use_bias=False,
            name="conv3",
        )
        self.bn3   = tf.keras.layers.BatchNormalization(name="bn3")
        self.act3  = tf.keras.layers.Activation("relu", name="relu3")
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool3")

        # ── Classificateur Dense ─────────────────────────────────────────────
        # Aplatit la feature map (4×4×128 = 2048) en vecteur 1D
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Couche Dense cachée
        self.dense1  = tf.keras.layers.Dense(256, activation="relu", name="dense1")

        # Dropout : régularisation (désactivé à l'inférence)
        self.dropout = tf.keras.layers.Dropout(0.5, name="dropout")

        # Couche de sortie : 10 neurones (un par classe), activation Softmax
        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="output"
        )

    def call(self, x, training: bool = False):
        """
        Propagation avant (forward pass).

        Paramètres
        ----------
        x        : tenseur d'entrée (batch, 32, 32, 3)
        training : bool — True pendant model.fit(), False pendant predict/evaluate
        """
        # Augmentation uniquement à l'entraînement
        if training:
            x = self.data_augmentation(x, training=training)

        # Bloc 1 : Conv → BN → ReLU → Pool
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)

        # Bloc 2 : Conv → BN → ReLU → Pool
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)

        # Bloc 3 : Conv → BN → ReLU → Pool
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)

        # Classificateur Dense
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def build_graph(self, input_shape=(32, 32, 3)):
        """Construit le graphe pour model.summary()."""
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x, training=False))
