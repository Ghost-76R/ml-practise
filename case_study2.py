#!/usr/bin/env python
# coding: utf-8

# In[60]:


#case study 1 (on small dataset)
# KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)


# In[61]:


cd C:\Users\Rohan Krishna Ullas\AppData\Local\Programs\Python\Python37


# In[62]:


x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
print(x,y)


# In[63]:


print(len(df1.columns))


# In[64]:


regressor=KNeighborsClassifier(n_neighbors=1)
sc1=StandardScaler()
#no need to scale y(dependent variable)
#feature scaling is important for KNN 
x=x.reshape((len(x),1))
y=y.reshape((len(y),1))
x=sc1.fit_transform(x)
print(x,y)
print(x.shape,y.shape)
regressor.fit(x,y)
y_pred=regressor.predict(x)
r2=r2_score(y,y_pred)
print('SCORE :'+str(r2))
plt.plot(x,y_pred,'r')
plt.scatter(x,y)
plt.show()
#most likely the the model is overfit


# In[65]:


help(regressor)


# In[ ]:




