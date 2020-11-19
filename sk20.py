#AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
df1=pd.read_csv('Mall_Customers.csv')
x=df1.iloc[:,[3,4]].values

dendrogram=hierarchy.dendrogram(hierarchy.linkage(x,method='ward'))
plt.show()
model1=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_pred1=model1.fit_predict(x)

plt.scatter(x[y_pred1==0,0],x[y_pred1==0,1],c='red',label='cluster1')
plt.scatter(x[y_pred1==1,0],x[y_pred1==1,1],c='orange',label='cluster2')
plt.scatter(x[y_pred1==2,0],x[y_pred1==2,1],c='yellow',label='cluster3')
plt.scatter(x[y_pred1==3,0],x[y_pred1==3,1],c='green',label='cluster4')
plt.scatter(x[y_pred1==4,0],x[y_pred1==4,1],c='blue',label='cluster5')
plt.legend()
plt.show()
