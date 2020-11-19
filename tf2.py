#normal equation implementation 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
df1=pd.DataFrame(data=fetch_california_housing().data,columns=fetch_california_housing().feature_names)
df1['target']=fetch_california_housing().target
print(df1.head())
sess1=tf.InteractiveSession()
m=len(df1.index)
n=len(df1.columns)-1
x=np.array(df1.iloc[:,:-1])

x=np.c_[np.ones((m,1)),x]
x=tf.constant(x)
y=np.array(df1['target']).reshape((-1,1))
y=tf.constant(y)
#inputs and outputs to nodes in a tf graph are tensors

xt=tf.transpose(x)
z=tf.matrix_inverse(tf.matmul(xt,x))
theta=tf.matmul(tf.matmul(z,xt),y)
init=tf.global_variables_initializer()

with tf.Session() as session1:
    init.run()
    print(theta.eval())
    result=tf.matmul(x,theta)
    print(result.eval())
    print(y.eval())
    print((result-y).eval())
    print('max error ={0}'.format(max(abs((result-y).eval()))))
