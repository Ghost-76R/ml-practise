#!/usr/bin/env python
# coding: utf-8

# In[3]:


#case study 4(on small dataset)
#LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)


# In[4]:


cd C:\Users\Rohan Krishna Ullas\AppData\Local\Programs\Python\Python37


# In[6]:


x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
print(x,y)
regressor=LinearRegression()
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




