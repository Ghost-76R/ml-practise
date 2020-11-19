#LinearRegression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df1=pd.read_csv('tvmarketing.csv')
#print(df1.head)
x=df1['TV']
y=df1['Sales']
plt.scatter(x,y)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=100)
#print(x_train.shape,"\n",x_test.shape,"\n",x.shape)
x_train=x_train[:,np.newaxis]
y_train=y_train[:,np.newaxis]
lr=LinearRegression()
lr.fit(x_train,y_train)
print("coef_ = "+str(lr.coef_)+"\nintercept = "+str(lr.intercept_))
x_test=x_test[:,np.newaxis]
y_pred=lr.predict(x_test)
s=[i for i in range(1,61)]
#s=s[:,np.newaxis]
plt.title('to compare y_pred and y_test')
plt.plot(s,y_pred,'r-')
plt.plot(s,y_test,'b-')
#sns.pairplot(x_vars='TV',y_vars='Sales',data=df1)
plt.show()
t1=mean_squared_error(y_test,y_pred)
t2=r2_score(y_test,y_pred)
print("MSS = "+str(t1)+"\nR^2 Score = "+str(t2))
plt.scatter(y_pred,y_test)
plt.title("ScatterPlot")
plt.show()
