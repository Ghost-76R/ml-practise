# multiple inputs and outputs
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Model
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

x_train_A=x_train.iloc[:,2:7]
x_train_B=x_train.iloc[:,:5]
inputA=keras.layers.Input(shape=x_train_A.shape[1:])
inputB=keras.layers.Input(shape=x_train_B.shape[1:])
hidden_layer1=keras.layers.Dense(units=50,kernel_initializer='uniform',activation='relu')(inputA)
hidden_layer2=keras.layers.Dense(units=30,kernel_initializer='uniform',activation='relu')(hidden_layer1)
concat=keras.layers.Concatenate()([hidden_layer2,inputB])
output=keras.layers.Dense(units=1,kernel_initializer='uniform',activation=None)(concat)
aux_output=keras.layers.Dense(units=1,kernel_initializer='uniform',activation=None)(hidden_layer2)
model1=Model(inputs=[inputA,inputB],outputs=[output,aux_output])

model1.compile(loss='mse',optimizer='adam')
hist=model1.fit([x_train_A,x_train_B],[y_train,y_train],epochs=20,batch_size=10,validation_split=0.1)

import matplotlib.pyplot as plt
df2=pd.DataFrame(data=hist.history)
df2.plot()
plt.show()
x_test_A=x_test.iloc[:,2:7]
x_test_B=x_test.iloc[:,:5]
model1.evaluate([x_test_A,x_test_B],[y_test,y_test])
