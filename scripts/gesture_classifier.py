"""
Gesture Classifier (MLP)
========================
A Multi-Layer Perceptron trained on MediaPipe hand landmark vectors.
Input: 63 normalized values (21 landmarks × [x, y, z])
Output: Gesture class label

Architecture:
  Input(63) → Dense(256) → BN → ReLU → Dropout(0.3)
            → Dense(128) → BN → ReLU → Dropout(0.2)
            → Dense(64)  → ReLU
            → Output(N_classes) → Softmax
"""

import numpy as np
import os
import json

# Optional: Use TensorFlow/Keras if available, else use sklearn
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


GESTURE_LABELS = [
    "move",
    "left_click",
    "right_click",
    "double_click",
    "scroll_up",
    "scroll_down",
    "drag",
    "volume_control",
    "stop"
]


class GestureClassifier:
    """
    Gesture classifier using MLP on MediaPipe landmark features.
    Falls back to sklearn if TensorFlow is not installed.
    """

    MODEL_PATH_TF = "models/gesture_model.h5"
    MODEL_PATH_SK = "models/gesture_model_sklearn.pkl"
    SCALER_PATH = "models/scaler.pkl"
    LABELS_PATH = "models/labels.json"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.labels = GESTURE_LABELS
        self.use_tf = TF_AVAILABLE
        self.loaded = False

    def build_keras_model(self, n_classes):
        """Build the MLP model with Keras."""
        model = Sequential([
            Dense(256, input_shape=(63,), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(n_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X, y, epochs=50, batch_size=32):
        """
        Train the gesture classifier.

        Args:
            X: np.array shape (N, 63) — normalized landmark vectors
            y: list of string labels
        """
        os.makedirs("models", exist_ok=True)

        # Encode labels
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        unique_labels = sorted(set(y))
        self.labels = unique_labels
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y_idx = np.array([label_to_idx[l] for l in y])

        # Save labels
        with open(self.LABELS_PATH, "w") as f:
            json.dump(unique_labels, f)

        # Scale features
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            joblib.dump(self.scaler, self.SCALER_PATH)
        else:
            X_scaled = X

        print(f"\n{'='*50}")
        print(f"Training Gesture Classifier")
        print(f"Samples: {len(X)} | Classes: {len(unique_labels)}")
        print(f"Labels: {unique_labels}")
        print(f"Backend: {'TensorFlow/Keras' if self.use_tf else 'scikit-learn'}")
        print(f"{'='*50}\n")

        if self.use_tf:
            n_classes = len(unique_labels)
            y_cat = to_categorical(y_idx, n_classes)

            self.model = self.build_keras_model(n_classes)
            self.model.summary()

            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
            ]
            history = self.model.fit(
                X_scaled, y_cat,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            self.model.save(self.MODEL_PATH_TF)
            print(f"\n✅ Keras model saved to {self.MODEL_PATH_TF}")
            return history

        elif SKLEARN_AVAILABLE:
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2,
                verbose=True
            )
            self.model.fit(X_scaled, y_idx)
            joblib.dump(self.model, self.MODEL_PATH_SK)
            print(f"\n✅ sklearn model saved to {self.MODEL_PATH_SK}")
            return None
        else:
            raise ImportError("Install tensorflow or scikit-learn to train.")

    def load(self):
        """Load a previously trained model."""
        os.makedirs("models", exist_ok=True)

        # Load labels
        if os.path.exists(self.LABELS_PATH):
            with open(self.LABELS_PATH) as f:
                self.labels = json.load(f)

        # Load scaler
        if SKLEARN_AVAILABLE and os.path.exists(self.SCALER_PATH):
            self.scaler = joblib.load(self.SCALER_PATH)

        # Try TF first
        if self.use_tf and os.path.exists(self.MODEL_PATH_TF):
            self.model = load_model(self.MODEL_PATH_TF)
            self.loaded = True
            print("✅ Loaded Keras model")
            return True

        # Try sklearn
        if SKLEARN_AVAILABLE and os.path.exists(self.MODEL_PATH_SK):
            self.model = joblib.load(self.MODEL_PATH_SK)
            self.use_tf = False
            self.loaded = True
            print("✅ Loaded sklearn model")
            return True

        print("⚠️  No trained model found. Run train.py first.")
        return False

    def predict(self, landmarks_vector):
        """
        Predict gesture from a 63-element landmark vector.

        Returns: (label: str, confidence: float)
        """
        if not self.loaded or self.model is None:
            return None, 0.0

        X = np.array(landmarks_vector).reshape(1, -1)
        if self.scaler:
            X = self.scaler.transform(X)

        if self.use_tf:
            probs = self.model.predict(X, verbose=0)[0]
            idx = np.argmax(probs)
            confidence = float(probs[idx])
        else:
            probs = self.model.predict_proba(X)[0]
            idx = np.argmax(probs)
            confidence = float(probs[idx])

        label = self.labels[idx] if idx < len(self.labels) else "unknown"
        return label, confidence


def rule_based_predict(fingers, landmarks):
    """
    Fallback rule-based gesture recognizer.
    Works without any trained model.
    Returns gesture label string.
    """
    if not fingers or not landmarks:
        return "stop"

    from utils import euclidean_distance

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]

    d_im = euclidean_distance(index_tip, middle_tip)
    d_ti = euclidean_distance(thumb_tip, index_tip)

    rules = {
        (0, 1, 0, 0, 0): "move",
        (0, 0, 0, 0, 1): "scroll_up",
        (1, 0, 0, 0, 1): "scroll_down",
        (1, 1, 1, 1, 1): "volume_control",
        (0, 0, 0, 0, 0): "stop",
    }

    f_tuple = tuple(fingers)
    if f_tuple in rules:
        return rules[f_tuple]

    if f_tuple == (0, 1, 1, 0, 0):
        if d_im < 38:
            return "left_click"
        return "move"

    if f_tuple == (1, 1, 0, 0, 0) and d_ti < 40:
        return "right_click"

    return "idle"