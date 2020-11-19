# saving and restoring models using callbacks
#callbacks used -> ModelCheckpoint,EarlyStopping and TensorBoard
#california housing dataset
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os,datetime

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
model1=keras.models.Sequential()
model1.add(keras.layers.Input(shape=(8)))
model1.add(keras.layers.Dense(units=10,input_dim=8,kernel_initializer='uniform',activation='relu'))
model1.add(keras.layers.Dense(units=5,kernel_initializer='uniform',activation='relu'))
model1.add(keras.layers.Dense(units=1,kernel_initializer='uniform',activation=None))

checkpoint=keras.callbacks.ModelCheckpoint('chkpnt1.h5',save_best_only=True)
early_stopping=keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
root_log_dir=os.path.join('logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback=keras.callbacks.TensorBoard(log_dir=root_log_dir)

model1.compile(loss='mse',optimizer='sgd')
hist=model1.fit(x_train,y_train,epochs=50,batch_size=10,validation_split=0.1,callbacks=[checkpoint,early_stopping,tensorboard_callback])
#help(model1.fit)

#model2=keras.models.load_model('chkpnt1.h5')

import matplotlib.pyplot as plt
df2=pd.DataFrame(data=hist.history)
df2.plot()
plt.show()


model1.evaluate(x_test,y_test)
model2=keras.models.load_model('chkpnt1.h5')
# here last checkpoint is decided by EarlyStopping callback
