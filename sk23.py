import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df1=pd.read_csv('Online+Retail.csv',encoding='ISO-8859-1')
print(df1.head)

#datapreprocessing techniques
df1.dropna(subset=['CustomerID'],inplace=True)
print(df1.isnull().sum())
print(df1.info())
df1['Amount']=df1['Quantity']*df1['UnitPrice']
cols=['Quantity','UnitPrice']
df1.drop(cols,axis=1,inplace=True)
x=df1.groupby('CustomerID').Amount.sum()
y=df1.groupby('CustomerID').InvoiceNo.count()
x=pd.merge(x,y,how='inner',on='CustomerID')
x.columns=['Amount','Frequency']
print(df1['InvoiceDate'].dtype)
df1['InvoiceDate']=pd.to_datetime(df1['InvoiceDate'])
print(df1['InvoiceDate'].dtype)
max_date=max(df1['InvoiceDate'])
print(type(max_date))
df1['Recency']=max_date-df1['InvoiceDate']
df1['dates']=df1['Recency'].dt.total_seconds()
#df1=df1.groupby('CustomerID').dates
print(df1.head())
z=df1[['CustomerID','dates']]
z=z.groupby('CustomerID').dates.min()
#z=df1.groupby('CustomerID').Recency
x=pd.merge(x,z,on='CustomerID',how='inner')
print(x.head())

#removing outliers
print(x.Amount.quantile(0.25),x.Amount.quantile(0.75))
t1=x.Amount.quantile(0.75)
t2=x.Amount.quantile(0.25)
d=t1-t2
outliers1=x[x['Amount']<(t1-1.5*d)]
outliers2=x[x['Amount']>(t1+1.5*d)]
x=x[x['Amount']>=(t1-1.5*d)]
x=x[x['Amount']<=(t1+1.5*d)]
print('mean = ',x.Amount.mean())
print(len(x.index))
print(outliers1['Amount'],outliers2['Amount'])

scaler=StandardScaler()
x=scaler.fit_transform(x)
x=pd.DataFrame(data=x,columns=['Amount','Frequency','dates'])
print(x)

model1=KMeans(n_clusters=5,max_iter=100,init='k-means++',n_init=10,random_state=100)
y_pred=model1.fit_predict(x)
print(y_pred)
modelX=GridSearchCV(estimator=model1,param_grid={'n_clusters':range(3,10)},cv=5)
modelX.fit(x)
print(modelX.best_params_)
#use elbow method over grid search
#check hopkins test and silhouette analysis

model4=KMeans(n_clusters=9,max_iter=100,init='k-means++',n_init=10,random_state=100)
model4.fit(x)
print(model4.inertia_)

wcss=[]
for i in range(3,10):
    model2=KMeans(n_clusters=i,max_iter=100,n_init=10,init='k-means++',random_state=100)
    model2.fit(x)
    wcss.append(model2.inertia_)
plt.plot(range(3,10),wcss,'r-')
plt.show()
