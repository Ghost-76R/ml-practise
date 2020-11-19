#use of scikit learn wrapper class
#california housing dataset
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

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

def build_model(units=30,n_layers=2,activation='relu',kernel_initializer='uniform'):
    model1=keras.models.Sequential()
    dict={'input_shape':[8]}
    for layer in range(n_layers):
        model1.add(keras.layers.Dense(units=30,activation=activation,kernel_initializer=kernel_initializer,**dict))
        dict={}
    model1.add(keras.layers.Dense(units=1,activation=None,kernel_initializer='uniform',**dict))
    model1.compile(optimizer='adam',loss='mse')
    return model1
model2=keras.wrappers.scikit_learn.KerasRegressor(build_model)
early_stopping=keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

model2.fit(x_train,y_train,epochs=50,validation_split=0.1,callbacks=[early_stopping])
print('score :{0}'.format(model2.score(x_test,y_test)))
y_pred=model2.predict(x_test)
#score will be negative of mse
print(mean_squared_error(y_test,y_pred))


modelY=keras.wrappers.scikit_learn.KerasRegressor(build_model,epochs=100,validation_split=0.1,callbacks=[early_stopping])
params={'units':[1,2,3],'n_layers':[1,2,3,4,5]}
#print(help(RandomizedSearchCV))
model3=RandomizedSearchCV(estimator=modelY,param_distributions=params,n_iter=10,cv=3)
model3.fit(x_train,y_train)
print(model3.best_params_)
