# to predict whether credit card users churn based on credit card usage
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler

df1=pd.read_csv('Churn_Modelling.csv')
print(df1.head())
labelencoder=LabelEncoder()
df1.iloc[:,5]=labelencoder.fit_transform(df1.iloc[:,5])
df2=pd.get_dummies(df1['Geography'],drop_first=True)
df1=pd.concat([df1,df2],axis=1)
df1.drop(['CustomerId','Surname','Geography','RowNumber'],axis=1,inplace=True)
print(df1.head())

y=df1['Exited']
x=df1.drop(['Exited'],axis=1)
scaler=StandardScaler()
cols=x.columns
x=scaler.fit_transform(x)
x=pd.DataFrame(data=x,columns=cols)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

model1=Sequential()
model1.add(Dense(input_dim=11,units=7,activation='relu',kernel_initializer='uniform'))
model1.add(Dropout(p=0.1))
model1.add(Dense(units=7,activation='relu',kernel_initializer='uniform'))
model1.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model1.fit(x_train,y_train,epochs=100,batch_size=10)

result=model1.predict(scaler.transform(np.array([[600,1,40,3,60000,2,1,1,50000,0,0]])))
print(result)

def build_func():
    #creating ann architecture
    model1=Sequential()
    model1.add(Dense(input_dim=11,output_dim=7,activation='relu',init='uniform'))
    model1.add(Dense(output_dim=7,activation='relu',init='uniform'))
    model1.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))
    model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model1
model2=KerasClassifier(build_fn=build_func,batch_size=10,epochs=100)
model2.fit(x_train,y_train)

scores=cross_val_score(estimator=model2,X=x_train,y=y_train,cv=10,n_jobs=-1)
print(scores)

params={'epochs':[10,100,300],'batch_size':[10,25,50]}
modelX=GridSearchCV(estimator=model2,param_grid=params,cv=10,scoring='accuracy')
modelX.fit(x_train,y_train)
print(modelX.best_params_,modelX.best_score_)
