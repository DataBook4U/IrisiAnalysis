"""
The data.py loads and performs changes on the dataset
"""

from sklearn.datasets import load_iris                      #Stellt Iris Datensatz bereit
from sklearn.model_selection import train_test_split        #Aufteilung in Test u. Trainingsdaten
from sklearn.preprocessing import StandardScaler            #Ermöglicht Standardisierung der Features


class DataLoader:
    def __init__(self):
        self.data = load_iris(as_frame=True)                #Datensatz wird als Pandas DataFrame geladen
        self.X = self.data.data                             #Klassifizierungs-Features
        self.y = self.data.target                           #Zielklasse

#Aufteilung Trainings- (80%) und Testdaten (20%)
    def get_splits(self, test_size=0.2, scale=True):                                    #scale=True -> skaliert Features mit StandardScaler
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
