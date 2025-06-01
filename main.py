import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")

print(iris.head())

sns.scatterplot (x='sepal length')
iris_setosa = iris["variety"]

print(iris_setosa.head())

print('Test3')