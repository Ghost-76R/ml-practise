#Feature Selection
""" feature selection can be done in three ways
1. filter method
2. wrapper method
3. embedded method
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.datasets import load_boston
df1=pd.DataFrame(data=load_boston().data,columns=load_boston().feature_names)
print(type(df1))
df1['target']=load_boston().target
for i in load_boston().keys():
    print(i)
x=df1.iloc[:,:-1]
y=df1.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=100)
x_train=sm.add_constant(x_train)
model1=sm.OLS(y_train,x_train).fit()
t=model1.pvalues
cols=list(t.index)
t=list(t)
print(cols)
#forward selection
for i in range(len(cols)):
    max_val=max(t)
    if(max_val>0.05):
        print(cols[t.index(max_val)])
        cols[t.index(max_val)]=0
        print(t.index(max_val))
        t.remove(max_val)
    else:
        break
cols.remove(0)
print(t,cols)
