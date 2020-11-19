#KMeans Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

df1=pd.read_csv('Mall_Customers.csv')
x=df1.iloc[:,[3,4]].values
kmeans1=KMeans(n_clusters=5,max_iter=100,init='k-means++',n_init=10,random_state=100)
y_pred=kmeans1.fit_predict(x)
print('WCSS = '+str(kmeans1.inertia_))

params={'n_clusters':[3,4,5,6,7,8,9,10],'init':['k-means++','random']}
modelX=GridSearchCV(estimator=kmeans1,param_grid=params,cv=5)
modelX.fit(x)
print(modelX.best_params_)
kmeans2=KMeans(n_clusters=10,init='random',n_init=10,max_iter=1000,random_state=100)
kmeans2.fit(x)
print('new WCSS ='+str(kmeans2.inertia_))

#elbow method to find best no: of clusters
wcss=[]
for i in [3,4,5,6,7,8,9,10]:
    model3=KMeans(n_clusters=i,max_iter=100,init='k-means++',n_init=10,random_state=100)
    model3.fit(x)
    wcss.append(model3.inertia_)
plt.plot(range(3,11),wcss,'r-')
plt.show()

print(y_pred)
# Visualising the clusters
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1] ,c = 'red', label = 'Cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], c = 'magenta', label = 'Cluster 5')
plt.legend()
plt.show()
