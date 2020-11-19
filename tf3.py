#gradient descent
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
initial_session=tf.InteractiveSession()
#to set it as default Session()
df1=pd.DataFrame(data=fetch_california_housing().data,columns=fetch_california_housing().feature_names)
df1['target']=fetch_california_housing().target
print(df1.head())

cols=df1.columns
scaler=StandardScaler()
df1=scaler.fit_transform(df1)
df1=pd.DataFrame(data=df1,columns=cols)
m=len(df1.index)
n=len(df1.columns)-1
x=np.array(df1.iloc[:,:-1])


x=np.c_[np.ones((m,1)),x]
x=tf.constant(x,dtype=tf.float32)
y=np.array(df1['target']).reshape((-1,1))
y=tf.constant(y,dtype=tf.float32)


epochs=1000
learning_rate=0.01
theta_val=tf.random_uniform([n+1,1],-1,1)
theta=tf.Variable(theta_val,name='theta',dtype=tf.float32)
y_pred=tf.matmul(x,theta,name='predictions')
error=y_pred-y
#error is also a tensor implying when inputs are tensors outputs are also tensors
mse=tf.reduce_mean(tf.square(error),name='mse')
gradients=tf.gradients(mse,[theta])[0]
modify=tf.assign(theta,(theta-(learning_rate*gradients)))
init=tf.global_variables_initializer()
initial_session.run(init)
print('initial :',gradients.eval())

#execution phase
with tf.Session() as session1:
    session1.run(init)
    for epoch in range(epochs):
        if(epoch%10==0):
            print('mse = {0}'.format(mse.eval()))
        modify.eval()
        #sess1.run(modify)
    print('theta :\n',theta.eval())
    print('predictions :',y_pred.eval())
    print('y :',y.eval())
    print('error :',error.eval())
    print('max error ={0}'.format(max(abs(error.eval()))))
