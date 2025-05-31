import pandas as pd

iris = pd.read_csv("iris.csv")

print(iris.head())

iris_setosa = iris["variety"]

print(iris_setosa.head())

print('Test3')