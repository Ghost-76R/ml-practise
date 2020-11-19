#RandomForestClassifier
#image classification
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
digits=load_digits()
print(type(digits.data),type(digits.target))
plt.figure(figsize=(20,4))
help(plt.figure)
print(digits.data.shape,"\t",digits.target.shape)
for i,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(3,5,i+1)
    print(type(label))
    #plt.imshow(image.reshape((8,8)),cmap=plt.cm.gray)
    #plt.show()
    print('Training:{0}\n '.format(label))
"""
np1=digits.data[9].reshape((8,8))
plt.imshow(np1,cmap=plt.cm.gray)
plt.show()
"""
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,train_size=0.75,random_state=100)
lm1=RandomForestClassifier(n_estimators=300)
lm1.fit(x_train,y_train)
"""
y_pred=lm1.predict(x_test)
for y,(image,label) in zip(y_pred[0:5],x_test):
    plt.subplot()
    plt.imshow(image.reshape((8,8)),cmap=plt.cm.gray)
    plt.show()
    print(" PREDICTED OUTPUT : {0}".format(y))
"""
for image,label in zip(digits.data[1700:1705],digits.target[1700:1705]):
    plt.subplot()
    y=lm1.predict(image.reshape(1,-1))
    plt.imshow(image.reshape((8,8)),cmap=plt.cm.gray)
    plt.show()
    print(" PREDICTED OUTPUT : {0}".format(y))
    print(" CORRECT OUTPUT : {0}".format(label))
print('Score = '+str(lm1.score(x_test,y_test)))
y_pred=lm1.predict(x_test)
print(accuracy_score(y_test,y_pred))
