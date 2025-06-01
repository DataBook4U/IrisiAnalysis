import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
df = iris


print(iris.head())

sns.scatterplot(x='sepal.length', y='sepal.width', style='variety', data=df)


plt.show()