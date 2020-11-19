# dimensionality reduction techniques
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score

df1=pd.read_csv('wine.csv')
print(df1.head,'\n',df1.shape)
x=df1.iloc[:,:-1]
y=df1.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
"""
pca1=PCA(n_components=2)
pca1.fit(x_train)
x_train_transform=pca1.transform(x_train)
print(type(x_train_transform))

pca2=KernelPCA(n_components=2,kernel='rbf')
pca2.fit(x_train)
x_train_transform=pca2.transform(x_train)
print(type(x_train_transform))
"""
#help(LinearDiscriminantAnalysis)
lda=LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_train,y_train)
x_train_transform=lda.transform(x_train)
print(type(x_train_transform))

x_train_transform=pd.DataFrame(data=x_train_transform)
print(len(x_train_transform.columns))
print(lda.explained_variance_ratio_.sum())

classifier1=RandomForestClassifier(n_estimators=100,random_state=100,max_depth=5,criterion='entropy')
classifier1.fit(x_train,y_train)
classifier2=LogisticRegression(solver='liblinear',random_state=100)
classifier2.fit(x_train,y_train)

y_pred1=classifier1.predict(x_test)
cm1=confusion_matrix(y_test,y_pred1)
print(cm1)
y_pred2=classifier2.predict(x_test)
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)

print('classifier1 ')
print('score :'+str(classifier1.score(x_test,y_test)))
print('cross_val_score : ',cross_val_score(classifier1,x_train,y_train,cv=5))
print('recall :'+str(recall_score(y_test,y_pred1,average='macro')))
print('precision :'+str(precision_score(y_test,y_pred1,average='macro')))

print('classifier2 ')
print('score :'+str(classifier2.score(x_test,y_test)))
print('cross_val_score : ',cross_val_score(classifier2,x_train,y_train,cv=5))
print('recall :'+str(recall_score(y_test,y_pred2,average='weighted')))
print('precision :'+str(precision_score(y_test,y_pred2,average='weighted')))
