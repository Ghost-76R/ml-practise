#!/usr/bin/env python
# coding: utf-8

# In[20]:


cd C:\Users\Rohan Krishna Ullas\AppData\Local\Programs\Python\Python37


# In[21]:


#case study 5(on small dataset)
# SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)


# In[24]:


x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
x=x.reshape((len(x),1))
y=y.reshape((len(y),1))
sc1=StandardScaler()
x=sc1.fit_transform(x)
print(x,y)
regressor=SVR(kernel='linear')
regressor.fit(x,y)
y_pred=regressor.predict(x)
r2=r2_score(y,y_pred)
print('SCORE :'+str(r2))
plt.plot(x,y_pred,'r')
plt.scatter(x,y)
plt.show()
# log :- dont know why completely different r2_score for svr model, shouldnt happen check for errors


# In[23]:


help(SVR)


# In[ ]:




