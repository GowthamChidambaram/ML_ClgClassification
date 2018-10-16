import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

#printing some basic information about the data set
df=pd.read_csv("College_Data",index_col=0)
print(df.head())
print(df.describe())
print(df.info())

#data visuals
sns.scatterplot(df["Grad.Rate"],df["Room.Board"],hue=df["Private"])
plt.show()
sns.scatterplot(df["F.Undergrad"],df["Outstate"],hue=df["Private"])
plt.show()
plt.figure(figsize=(10,6))
df[df["Private"]=="Yes"]["Outstate"].hist(bins=35,alpha=0.7)
df[df["Private"]=="No"]["Outstate"].hist(bins=35,alpha=0.7)
plt.show()
plt.figure(figsize=(10,6))
df[df["Private"]=="Yes"]["Grad.Rate"].hist(bins=35,alpha=0.7)
df[df["Private"]=="No"]["Grad.Rate"].hist(bins=35,alpha=0.7)
plt.show()

#data clean
print(df[df["Grad.Rate"]>100])
df["Grad.Rate"]["Cazenovia College"] = 100
print(df[df["Grad.Rate"]>100])
plt.figure(figsize=(10,6))
df[df["Private"]=="Yes"]["Grad.Rate"].hist(bins=35,alpha=0.7)
df[df["Private"]=="No"]["Grad.Rate"].hist(bins=35,alpha=0.7)
plt.show()

#k_means
kmeans=KMeans(n_clusters=2)
kmeans.fit(df.drop("Private",axis=1))
print(kmeans.labels_)
print(kmeans.cluster_centers_)
def converter(k):
    if k=="Yes":
        return 1
    else:
        return 0
df["Cluster"]=df["Private"].apply(converter)
print(df[["Private","Cluster"]])

#evaluating
print("confusion matrix : ")
print(confusion_matrix(kmeans.labels_,df["Cluster"]))
print("Classification report :")
print(classification_report(kmeans.labels_,df["Cluster"]))







