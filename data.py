"""
The data.py loads and performs changes on the dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self):
        self.data = load_iris(as_frame=True)
        self.X = self.data.data
        self.y = self.data.target

    def get_splits(self, test_size=0.2, scale=True):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
