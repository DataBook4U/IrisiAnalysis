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

    def __init__(self, DataFrame, DataFrameColumns):
        self.DataFrame = DataFrame
        self.DataFrameColumns = DataFrameColumns
        self.average = None     #to get average value of a column
        self.HasNull = False    #to get amount of Null values in a column

    #Berechne den Mittelwert für eine Spalte im Datensatz:
    def GetAverageAll(self):
        for column in self.DataFrameColumns[:-1]:                                     #the last element is the target column that I don't want to look at for now
            self.average = self.DataFrame[column].mean()
            print("Average value of " + column + " : " + str(self.average))

    #Gib mir an wie viele Null Werte jeweils in meinen Spalten enthalten sind:
    def GetNullAll(self):
        for column in self.DataFrameColumns:
            null_count = self.DataFrame[column].isnull().sum()
            if null_count > 0:
                print("The column " + column + " has " + str(null_count) + " NULL values.")
                self.HasNull = True
            else:
                self.HasNull = False
        if not self.HasNull:                #My version was: if self.HasNull == False:
                print("There are no missing values in your DataFrame")





#Load Iris DataSet from Library
iris = load_iris()

#Use transformer and create DF
transformer = TransformData(iris)
df_iris = transformer.ToDF()
columns = transformer.ColumnsDF()
print(columns)
#print(transformer.df)

#Get Data Insights
test1 = ExploreData(df_iris, columns)
test1.GetNullAll()

#Create a function that goes through the DF by all the columns and prints out their average values


#Would be cool to have a script that goes through a chosen dataset and tells you:
#   *If Null Values in DataSet and how many
#   *The Mean Values of