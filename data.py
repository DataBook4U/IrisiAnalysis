"""
The data.py loads and performs changes on the dataset
"""
import pandas as pd
from sklearn.datasets import load_iris                      #Stellt Iris Datensatz bereit
from sklearn.model_selection import train_test_split        #Aufteilung in Test u. Trainingsdaten
from sklearn.preprocessing import StandardScaler            #Ermöglicht Standardisierung der Features


#Maybe move to seperate file like "explore.py"??

class TransformData:

    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None

    def ToDF(self):
        self.df = pd.DataFrame(data=self.dataset.data, columns=self.dataset.feature_names)
        self.df["target"] = self.dataset.target
        return self.df

    def ShowHead(self):
        if self.df is not None:
            return self.df.head()
        else:
            return "DataFrame was not created!"


#Load Iris DataSet from Library
iris = load_iris()

#Use transformer and create DF
transformer = TransformData(iris)
df_iris = transformer.ToDF()
print(transformer.ShowHead())
print(transformer.df)

