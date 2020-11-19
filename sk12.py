#RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)
x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
#not splitting into training and testing sets
x=x.reshape(len(x),1)
y=y.reshape(len(y),1)
regressor=RandomForestRegressor(n_estimators=100,random_state=100,bootstrap=True)
regressor.fit(x,y)
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
y_pred=regressor.predict(x_grid)
plt.scatter(x,y)
plt.plot(x_grid,y_pred,'r-')
plt.show()
print(r2_score(regressor.predict(x),y))
x_new=np.array([[6.5]])
x_new=x_new.reshape(len(x_new),1)
print(regressor.predict(x_new))
