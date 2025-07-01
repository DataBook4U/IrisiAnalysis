"""
The models.py loads the ml models
"""


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNModel:
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)

    def predict(self, X_new):
        return self.model.predict(X_new)