#!/usr/bin/env python
# coding: utf-8

# In[6]:


# case study 3(on small dataset)
#LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)


# In[7]:


cd C:\Users\Rohan Krishna Ullas\AppData\Local\Programs\Python\Python37


# In[9]:


x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
print(x,y)
regressor=LogisticRegression(solver='lbfgs')
x=x.reshape((len(x),1))
y=y.reshape((len(y),1))
regressor.fit(x,y)
y_pred=regressor.predict(x)
r2=r2_score(y,y_pred)
print('SCORE :'+str(r2))

plt.plot(x,y_pred,'r')
plt.scatter(x,y)
plt.show()


# In[ ]:




