#Functional API
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

df1=pd.DataFrame(data=fetch_california_housing().data,columns=fetch_california_housing().feature_names)
df1['target']=fetch_california_housing().target
print(df1.head())

scaler=StandardScaler()
#x_train_transform=scaler.fit_transform(x_train)
#x_test=scaler.transform(x_test)

df1=pd.DataFrame(data=scaler.fit_transform(df1))

x=df1.iloc[:,:-1]
y=df1.iloc[:,-1]
print(x.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

input_layer=keras.layers.Input(shape=x.shape[1:])
hidden1=keras.layers.Dense(units=30,activation='relu',kernel_initializer='uniform')(input_layer)
hidden2=keras.layers.Dense(units=10,activation='relu',kernel_initializer='uniform')(hidden1)
concat=keras.layers.Concatenate()([input_layer,hidden2])
output_layer=keras.layers.Dense(units=1,activation=None)(concat)
model1=keras.models.Model(inputs=[input_layer],outputs=[output_layer])
#Model class is similiar to Sequential tensorflow.keras.models.Model  tensorflow.keras.models.Sequential

model1.compile(loss='mse',optimizer='adam')
hist=model1.fit(x_train,y_train,epochs=10,batch_size=10,validation_split=0.1)
model1.evaluate(x_test,y_test)

import matplotlib.pyplot as plt
df2=pd.DataFrame(data=hist.history)
df2.plot()
plt.show()
