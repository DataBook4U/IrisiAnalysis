"""
The data.py loads and performs changes on the dataset
"""
import pandas as pd
from sklearn.datasets import load_iris                      #Stellt Iris Datensatz bereit
from sklearn.model_selection import train_test_split        #Aufteilung in Test u. Trainingsdaten
from sklearn.preprocessing import StandardScaler            #Ermöglicht Standardisierung der Features


#Maybe move to seperate file like "explore.py"??

iris = load_iris()

df = pd.DataFrame(data=iris, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())