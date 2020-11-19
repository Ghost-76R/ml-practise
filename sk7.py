#polynomial regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)
x=df1.iloc[:,[1]]
y=df1.iloc[:,[2]]
print(type(x),type(y))
#not splitting into training and testing sets

#creating LinearRegression model1
#fitting linear regression to dataset
lm1=LinearRegression().fit(x,y)
y_pred1=lm1.predict(x)

#creating LinearRegression model2
#fitting polynomial regression to dataset
x_poly=PolynomialFeatures(degree=5).fit_transform(x)
print(x_poly)
lm2=LinearRegression()
lm2.fit(x_poly,y)
y_pred2=lm2.predict(x_poly)
help(PolynomialFeatures)
#visualising LinearRegression results
plt.plot(x,y_pred1,'r-')
plt.scatter(x,y)
plt.title('LinearRegression')
plt.show()

#visualising polynomialregression results
plt.plot(x,y_pred2,'r-')
plt.scatter(x,y)
plt.title('PolynomialRegression')
plt.show()
