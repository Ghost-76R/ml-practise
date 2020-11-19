import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix

df1=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
print(df1.head)
#print(stopwords.words('english'))

for i in range(len(df1.index)):
    text=df1.iloc[i,0]
    text=re.sub('[^a-zA-Z]',' ',text)
    text=(text.lower()).split()
    ps=PorterStemmer()
    print(text)
    text=[ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    print(text)
    text=' '.join(text)
    df1.iloc[i,0]=text
vectorizer=CountVectorizer(max_features=1500,lowercase=True)
x=df1.iloc[:,0]
y=df1.iloc[:,1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
x_train_transform=vectorizer.fit_transform(x_train).toarray()
x_test=vectorizer.transform(x_test).toarray()

model1=BernoulliNB()
model1.fit(x_train_transform,y_train)

model2=RandomForestClassifier(n_estimators=300,criterion='gini',random_state=100,max_depth=5)
model2.fit(x_train_transform,y_train)

modelX=XGBClassifier(eta=0.1,silent=1,objective='binary:logistic',eval_metric='error',random_state=100,n_estimators=300)
modelX.fit(x_train_transform,y_train)

print(model1.score(x_test,y_test))
print(model2.score(x_test,y_test))
print(modelX.score(x_test,y_test))
print(confusion_matrix(y_test,model1.predict(x_test)))
print(confusion_matrix(y_test,model2.predict(x_test)))
print(confusion_matrix(y_test,modelX.predict(x_test)))
