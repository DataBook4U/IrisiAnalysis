"""
The main.py is intended for loading data, training models, analysing models and testing predictions
"""

import pandas as pd
from data import DataLoader
from models import KNNModel

def main():
    # Daten laden
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.get_splits()

    # Modell trainieren
    knn = KNNModel(n_neighbors=5)
    knn.train(X_train, y_train)

    # Auswertung
    accuracy = knn.evaluate(X_test, y_test)
    print(f"KNN Genauigkeit: {accuracy:.2f}")

    # Prognose auf neuen Daten (optional)
    sample = [X_test[0]]  # Beispiel
    prediction = knn.predict(sample)
    print(f"Vorhersage für Sample: {prediction}")

if __name__ == "__main__":
    main()