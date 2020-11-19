# DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
df1=pd.read_csv('Position_Salaries.csv')
print(df1.head)
x=df1.iloc[:,1].values
y=df1.iloc[:,2].values
x=x.reshape((len(x),1))
y=y.reshape((len(y),1))
regressor=DecisionTreeRegressor(max_depth=2,random_state=100)
regressor.fit(x,y)
#print(y_new.shape)
x_new=np.arange(min(x),max(x),0.01)
x_new=x_new[:,np.newaxis]
plt.scatter(x,y)
plt.plot(x_new,regressor.predict(x_new),'r-')
plt.show()
y_pred=regressor.predict(x)
print(y_pred)
print(r2_score(y,y_pred))
print('Score ='+str(regressor.score(x,y)) )
#the model becomes overfit if max_depth=1 or greater than 2
