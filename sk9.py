#regression using SVR 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
df1=pd.read_csv('Position_Salaries.csv')
#print(df1)
x=df1.iloc[:,[1]].values
y=df1.iloc[:,[2]].values

sc1=StandardScaler()
sc2=StandardScaler()
x=sc1.fit_transform(x)
y=sc2.fit_transform(y)
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

regressor2=KNeighborsRegressor(n_neighbors=3)
regressor2.fit(x,y)

print(x,y)
y_pred_val=sc2.inverse_transform(regressor.predict(sc1.transform(np.array([[6.5]]))))
print(y_pred_val)
y_pred=regressor.predict(x)
y_pred2=regressor2.predict(x)
plt.plot(x,y,'r-')
plt.show()
plt.plot(sc1.inverse_transform(x),sc2.inverse_transform(y),'r-')
plt.show()

print('r2 score =',r2_score(y,y_pred))
print('MSS = ',mean_squared_error(y,y_pred))

print('r2 score =',r2_score(y,y_pred2))
print('MSS = ',mean_squared_error(y,y_pred2))

