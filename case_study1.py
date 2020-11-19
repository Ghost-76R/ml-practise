#!/usr/bin/env python
# coding: utf-8

# In[38]:


cd C:\Users\Rohan Krishna Ullas\AppData\Local\Programs\Python\Python37


# In[39]:


#case study 1 (on small dataset)
# DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)


# In[42]:


x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
x=x.reshape((len(x),1))
y=y.reshape((len(y),1))
regressor=DecisionTreeRegressor(random_state=100)
regressor.fit(x,y)
#print(y_new.shape)
"""
plt.scatter(x,y)
plt.plot(x_new,regressor.predict(x_new),'b-')
plt.show()
"""
y_pred=regressor.predict(x)
print('SCORE : '+str(r2_score(y,y_pred)))
plt.plot(x,y_pred,'r')
plt.scatter(x,y)
plt.show()
#most likely the the model is overfit


# In[41]:


help(regressor)


# In[ ]:




