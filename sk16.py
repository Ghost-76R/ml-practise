#SVM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
df1=pd.read_csv('Social_Network_Ads.csv')
print(df1.head)
labelencoder=LabelEncoder()
#help(labelencoder)
df1.iloc[:,[1]]=labelencoder.fit_transform(df1.iloc[:,1])
x=df1.iloc[:,[1,2,3]].values
y=df1.iloc[:,[4]].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=100,train_size=0.75)
#preprocessing
sc=StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)
#print(x_train,y_train,x_train.shape,y_train.shape)
#help(KNeighborsRegressor)
classifier1=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier1.fit(x_train,y_train)
y_pred=classifier1.predict(x_test)

classifier2=LogisticRegression(solver='lbfgs')
classifier2.fit(x_train,y_train)
y_pred2=classifier2.predict(x_test)

classifier3=SVC(kernel='rbf',random_state=100)
classifier3.fit(x_train,y_train)
y_pred3=classifier3.predict(x_test)
#help(SVC)
#creating a confusion matrix
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
cm3=confusion_matrix(y_test,y_pred3)
print(cm3)
print('Score1(KNeighborsClassifier) = '+str(classifier1.score(x_test,y_test)))
print('Score2(LogisticRegression)= '+str(classifier2.score(x_test,y_test)))
print('Score3(SVC with kernel=rbf) = '+str(classifier3.score(x_test,y_test)))
sns.heatmap(cm1,annot=True)
plt.show()
sns.heatmap(cm2,annot=True)
plt.show()
sns.heatmap(cm3,annot=True)
plt.show()
#print(classifier2.predict_proba(x_test),classifier2.predict(x_test))
