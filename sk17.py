# image classification on MNIST dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df1=pd.read_csv('mnist_train.csv')
df2=pd.read_csv('mnist_test.csv')
x_train=df1.iloc[:,1:]
y_train=df1.iloc[:,[0]]
x_test=df2.iloc[:,1:]
y_test=df2.iloc[:,[0]]
print(x_train.shape)
print(y_train.shape)
print(x_train.iloc[5,:].shape)
for i in range(10):
    image=np.array(x_train.iloc[i,:])
    plt.imshow(image.reshape(28,28),cmap=plt.cm.gray)
    plt.show()
    print(y_train.iloc[i,[0]])
#creating SVM classifier
classifier1=SVC(kernel='linear')
classifier1.fit(x_train.iloc[:5000,:],y_train.iloc[:5000,:])
y_pred1=classifier1.predict(x_test.iloc[:1000,:])

#RandomForestClassifier
classifier2=RandomForestClassifier(n_estimators=100,random_state=100)
classifier2.fit(x_train.iloc[:5000,:],y_train.iloc[:5000,:])
y_pred2=classifier2.predict(x_test.iloc[:1000,:])

#KNeighborsClassifier
classifier3=KNeighborsClassifier(n_neighbors=5)
classifier3.fit(x_train.iloc[:5000,:],y_train.iloc[:5000,:])
y_pred3=classifier3.predict(x_test.iloc[:1000,:])

params={'n_estimators':[100,150,200,250,300,350]}
model3=GridSearchCV(estimator=classifier2,param_grid=params,cv=5)
model3.fit(x_train.iloc[:5000,:],y_train.iloc[:5000,:])

print(model3.best_params_)
print(model3.best_estimator_)
print(cross_val_score(model3,x_train.iloc[:5000,:],y_train.iloc[:5000,:],cv=5))
print(y_pred1)

for i in range(50,60):
    image=np.array(x_test.iloc[i,:])
    plt.imshow(image.reshape(28,28),cmap=plt.cm.gray)
    plt.show()
    pred_label=y_pred1[i]
    print('predicted label :{0}'.format(pred_label))

print('score :',classifier1.score(x_test.iloc[:1000,:],y_test.iloc[:1000,:]))
print('score :',classifier2.score(x_test.iloc[:1000,:],y_test.iloc[:1000,:]))
print('score :',classifier3.score(x_test.iloc[:1000,:],y_test.iloc[:1000,:]))
