import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
#text classification problem1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
df1=pd.read_csv('example_train1.csv')
print(df1)
x_train=df1.iloc[:,0]
lb=LabelEncoder()
lb.fit(df1['Class'])
df1['Class']=lb.transform(df1['Class'])
#df1['Class']=df1['Class'].map({'education':0,'cinema':1})
print(df1)
y_train=df1.iloc[:,1]

vectorizer=CountVectorizer(stop_words='english')
vectorizer.fit(x_train)
print(vectorizer.vocabulary_)
#print(vectorizer.get_feature_names)
x=vectorizer.transform(x_train)

x=x.toarray()
print(type(x),'\n',x)
print(type(vectorizer.vocabulary_))
ls=[]
for i in vectorizer.vocabulary_.keys():
    ls.append(i)
print(ls)
df2=pd.DataFrame(data=x,columns=ls)
df2['Label']=df1['Class']
print(df2)

x_train_new=df2.iloc[:,:-1]
y_train_new=df1.iloc[:,-1]
model1=MultinomialNB()
model1.fit(x_train_new,y_train_new)

model2=BernoulliNB()
model2.fit(x_train_new,y_train_new)

y_pred_proba=model1.predict_proba(x_train_new)
print(y_pred_proba)
print(model2.predict_proba(x_train_new))
