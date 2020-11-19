#spam classification
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

df1=pd.read_table('SMSSpamCollection+(1)',names=['label','class'])
print(df1.shape,df1.head())
x=df1.iloc[:,1]
y=df1.iloc[:,0]
lb=LabelEncoder()
lb.fit(y)
y=lb.transform(y)
y=y.T
print(y,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

vectorizer=CountVectorizer(stop_words='english')
vectorizer.fit(x_train)
x_train=vectorizer.transform(x_train)
print(vectorizer.vocabulary_,type(x_train))
print(x_train)

model1=MultinomialNB()
model1.fit(x_train,y_train)

df2=pd.DataFrame(data=x_test)
x_test=vectorizer.transform(x_test)
y_pred=model1.predict(x_test)
df2['label']=y_pred
df2.iloc[:,1]=df2.iloc[:,1].map({0:'ham',1:'spam'})
print(df2)

print('score ='+str(model1.score(x_test,y_test)))
cm=confusion_matrix(y_test,y_pred)
print(cm)
