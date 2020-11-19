import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm
#data preparation
df1=pd.read_csv('Housing.csv')
print(df1.head,"\n",df1.info)
def fun1(var):
    if var=='yes':
        return(1)
    else:
        return(0)
print(df1.columns)
def normalise(x):
    #print(type(x))
    return((x-min(x))/(max(x)-min(x)))
df1['prefarea']=df1['prefarea'].apply(fun1)
df1['mainroad']=df1['mainroad'].apply(fun1)
df1['guestroom']=df1['guestroom'].apply(fun1)
df1['prefarea']=df1['prefarea'].apply(fun1)
df1['basement']=df1['basement'].apply(fun1)
df1['airconditioning']=df1['airconditioning'].apply(fun1)
df1['hotwaterheating']=df1['hotwaterheating'].apply(fun1)
df2=pd.get_dummies(df1['furnishingstatus'])
print(df2)
df1=pd.concat([df1,df2],axis=1)
print(df1.columns)
df1.drop('furnished',axis=1,inplace=True)
df1.drop('furnishingstatus',axis=1,inplace=True)
df1['areaperbedroom']=df1['area']/df1['bedrooms']
df1['bathroomsperrbedroom']=df1['bathrooms']/df1['bedrooms']
print(df1)
#splitting data into training and testing sets
y=df1['price']
x=df1.drop('price',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=100)
lm1=LinearRegression()
rfe=RFE(lm1,9).fit(x_train,y_train)
cols=x_train.columns[rfe.support_]
x_train=x_train[cols]
x_test=x_test[cols]
# creating model using sklearn
lm2=LinearRegression()
lm2.fit(x_train,y_train)
y_pred=lm2.predict(x_test)
print('PREDICITONS \n',y_pred)
print('MSE = ',mean_squared_error(y_test,y_pred))
print('R^2 SCORE = ',r2_score(y_test,y_pred))
#r2_score is same or different from rsquared?
# creating OLS model
x_train=sm.add_constant(x_train)
x_test=sm.add_constant(x_test)
lm3=sm.OLS(y_train,x_train).fit()
print("OLS model \nrsquared = ",lm3.summary())
y_pred2=lm3.predict(x_test)
print("rsquared = "+str(lm3.rsquared))
print("MSE = ",mean_squared_error(y_test,y_pred2))
