#naive_baiyes classifier
#naive_bayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
from xgboost.sklearn import XGBClassifier

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

"""
pca=PCA(n_components=0.95)
cols=x_train.columns
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)

"""
classifier1=KNeighborsClassifier(n_neighbors=7,metric='minkowski',p=2)
classifier1.fit(x_train,y_train)
y_pred=classifier1.predict(x_test)

classifier2=LogisticRegression(solver='lbfgs')
classifier2.fit(x_train,y_train)
y_pred2=classifier2.predict(x_test)

classifier3=SVC(kernel='rbf',random_state=100)
classifier3.fit(x_train,y_train)
y_pred3=classifier3.predict(x_test)

modelX=GridSearchCV(estimator=classifier1,param_grid={'n_neighbors':[1,3,5,7,9]},cv=5)
modelX.fit(x_train,y_train)
print(modelX.best_params_)

classifier4=RandomForestClassifier(criterion='entropy',n_estimators=100,random_state=100)
classifier4.fit(x_train,y_train)
#scaling not really neccessary for RandomForestClassifier
y_pred4=classifier4.predict(x_test)

classifier5=GaussianNB()
classifier5.fit(x_train,y_train)
y_pred5=classifier5.predict(x_test)

classifier6=DecisionTreeClassifier(criterion='entropy',random_state=100)
classifier6.fit(x_train,y_train)
y_pred6=classifier6.predict(x_test)

#XGBCLassifier
classifier7=XGBClassifier(booster='gbtree',max_depth=5,subsample=0.8,reg_alpha=1,reg_lambda=1,random_state=100)
classifier7.fit(x_train.iloc[:5000,:],y_train.iloc[:5000,:])
y_pred7=classifier4.predict(x_test.iloc[:1000,:])

#help(SVC)
#creating a confusion matrix
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
cm3=confusion_matrix(y_test,y_pred3)
print(cm3)
cm4=confusion_matrix(y_test,y_pred4)
print(cm4)
cm5=confusion_matrix(y_test,y_pred5)
print(cm5)
cm6=confusion_matrix(y_test,y_pred6)
print(cm6)

print('Score1(KNeighborsClassifier) = '+str(classifier1.score(x_test,y_test)))
print('Score2(LogisticRegression)= '+str(classifier2.score(x_test,y_test)))
print('Score3(SVC with kernel=rbf) = '+str(classifier3.score(x_test,y_test)))
print('Score4(RandomForestClassifier) = '+str(classifier4.score(x_test,y_test)))
print('Score5(Ga ussianNB) = '+str(classifier5.score(x_test,y_test)))
print('Score6(DecisionTreeClassifier) = '+str(classifier6.score(x_test,y_test)))
"""
sns.heatmap(cm1,annot=True)
plt.show()
sns.heatmap(cm2,annot=True)
plt.show()
sns.heatmap(cm3,annot=True)
plt.show()
#print(classifier2.predict_proba(x_test),classifier2.predict(x_test))
"""
