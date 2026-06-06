import numpy as np
import pandas as pd
import matplotlib
(matplotlib.use('Agg'))
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
url = "https://raw.githubusercontent.com/rolandmueller/titanic/main/titanic3.csv"
df = pd.read_csv(url)
print(df.head())

#inspection and finding missing values

print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

#cleaning the data

df["age"] = df["age"].fillna(df["age"].mean())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode())
df = df.drop(columns=["cabin", "body", "boat"])

#visualisation

#bar graph to visualise survival and sex

plt.figure(figsize = (6,4))
plt.yticks(range(0, 851, 50))
sns.countplot(data=df, x="survived", hue="sex")
plt.savefig("survived.png")
plt.close()

#histogram for ages of the crowd

plt.figure(figsize = (6,4))
sns.histplot(data=df, x='age', bins=30, kde=True, hue="survived")
plt.savefig("age.png")
plt.close()

#correlation heatmap

plt.figure(figsize = (6,4))
numerical_df = df.select_dtypes(include=["float64", "int64"])
grid_data = numerical_df.corr()
sns.heatmap(data=grid_data  , annot=True, cmap="coolwarm")
plt.savefig("heatmap.png")
plt.close()