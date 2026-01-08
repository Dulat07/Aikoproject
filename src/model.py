import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def train_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Train a supervised RandomForest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    return model


def train_isolation_forest(X: pd.DataFrame):
    """
    Train an unsupervised anomaly detection model.
    """
    model = IsolationForest(
        contamination=0.10,  # 10% anomalies expected
        random_state=42
    )
    model.fit(X)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification accuracy.
    """
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def build_lstm(input_shape):
    """
    Build simple binary classification LSTM.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # same user = 1, impostor = 0
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_lstm(sequences, labels, epochs=20):
    """
    Train LSTM on sequence data.
    """
    X = np.array(sequences)
    y = np.array(labels)

    model = build_lstm((X.shape[1], X.shape[2]))

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=4,
        verbose=1
    )

    return model