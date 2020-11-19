#regression problem 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

print(tf.__version__)
dataset = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset)
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
df1=pd.read_csv(dataset,sep=" ",na_values = '?',names=column_names,comment='\t',skipinitialspace=True)
print(df1.head())
df1.isnull().sum()

df1=df1[~np.isnan(df1['Horsepower'])]
print(df1.isnull().sum())
print(df1.info())

df2=pd.get_dummies(df1['Origin'],drop_first=True)
df1=pd.concat([df1,df2],axis=1)
df1.drop('Origin',axis=1,inplace=True)
print(df1.head())

scaler1=StandardScaler()
y=df1['MPG']
x=df1.drop('MPG',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
cols=x_train.columns
x_train_new=scaler1.fit_transform(x_train)
x_train=pd.DataFrame(data=x_train_new,columns=cols)
x_test_new=scaler1.transform(x_test)
x_test=pd.DataFrame(data=x_test_new,columns=cols)

"""
import seaborn as sns
sns.pairplot(df1)
"""

early_stopping=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
#model 
model1=keras.models.Sequential()
model1.add(keras.layers.Input(shape=x_train.shape[1:]))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Dense(units=64,kernel_initializer='lecun_normal'))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Activation('selu'))
model1.add(keras.layers.Dense(units=64,kernel_initializer='lecun_normal',activation='selu'))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Activation('selu'))
#model1.add(keras.layers.Dense(units=32,kernel_initializer='lecun_normal',activation='selu'))

model1.add(keras.layers.Dense(units=1,activation=None))
optimizer=keras.optimizers.RMSprop(0.001)
model1.compile(loss='mse',metrics=['mse'],optimizer=optimizer)
model1.fit(x_train,y_train,epochs=1000,batch_size=10,validation_split=0.1,callbacks=[early_stopping])
print(len(x_train.index),len(y_train.index))

#model 2
model2=keras.models.Sequential()
model2.add(keras.layers.Input(shape=x_train.shape[1:]))
model2.add(keras.layers.Dense(units=64,kernel_initializer='lecun_normal'))
model2.add(keras.layers.Activation('selu'))
model2.add(keras.layers.Dense(units=64,kernel_initializer='lecun_normal',activation='selu'))
model2.add(keras.layers.Activation('selu'))
#model2.add(keras.layers.Dense(units=32,kernel_initializer='lecun_normal',activation='selu'))

model2.add(keras.layers.Dense(units=1,activation=None))
optimizer=keras.optimizers.RMSprop(0.001)
model2.compile(loss='mse',metrics=['mse'],optimizer='nadam')
model2.fit(x_train,y_train,epochs=1000,batch_size=10,validation_split=0.1,callbacks=[early_stopping])

model1.evaluate(x_test,y_test)
model2.evaluate(x_test,y_test)
