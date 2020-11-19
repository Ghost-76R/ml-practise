#LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
df1=pd.read_csv('Social_Network_Ads.csv')
print(df1.head)
#x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=100)
#print(df1.head)
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
print(x_train,y_train,x_train.shape,y_train.shape)

classifier=LogisticRegression(solver='lbfgs')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

classifier2=KNeighborsClassifier(n_neighbors=5)
classifier2.fit(x_train,y_train)
y_pred2=classifier2.predict(x_test)
cm3=confusion_matrix(y_test,y_pred2)

#creating a confusion matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(cm3)
print('Using LogisitcRegression\nScore = '+str(classifier.score(x_test,y_test)))
print('accuracy = '+str(accuracy_score(y_pred,y_test)))
sns.heatmap(cm,annot=True)
plt.show()

print('Using KNeigbhorsClassifier\nScore = '+str(classifier2.score(x_test,y_test)))
print('accuracy = '+str(accuracy_score(y_pred2,y_test)))
sns.heatmap(cm3,annot=True)
plt.show()
