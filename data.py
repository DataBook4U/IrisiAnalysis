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
        self.columns = None

    #Transforms original Dataset and gives back a pandas DataFrame
    def ToDF(self):
        self.df = pd.DataFrame(data=self.dataset.data, columns=self.dataset.feature_names)
        self.df["target"] = self.dataset.target
        return self.df

    #Gives back List of the DataFrame columns
    def ColumnsDF(self):
        self.columns = self.df.columns.tolist()
        return self.columns


class ExploreData:

    def __init__(self, DataFrame, DataColumn):
        self.DataFrame = DataFrame
        self.DataColumn = DataColumn
        self.average = None     #to get average value of a column
        self.HasNull = None     #to get amount of Null values in a column

    def GetAverage(self):
        self.average = self.DataFrame[self.DataColumn].mean()
        print("Average value of " + self.DataColumn + " : " + str(self.average))

    def GetAverageAll(self, columns):
        for column in columns:
            self.average = self.DataFrame[column].mean()
            print("Average value of " + column + " : " + str(self.average))

#Load Iris DataSet from Library
iris = load_iris()

#Use transformer and create DF
transformer = TransformData(iris)
df_iris = transformer.ToDF()
columns = transformer.ColumnsDF()
print(columns)
#print(transformer.df)

#Get Data Insights
test1 = ExploreData(df_iris, "petal length (cm)")
test1.GetAverageAll(columns)

#Create a function that goes through the DF by all the columns and prints out their average values
