#sentiment analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import f1_score,confusion_matrix,recall_score,precision_score
from sklearn.utils import resample

df1=pd.read_csv('train_tweets.csv')
df2=pd.read_csv('test_tweets.csv')
print(df1.head(),df2.head())
df1_resample1=df1[df1['label']==0]
df1_resample2=df1[df1['label']==1]

#over-sampling
df1_resample2=resample(df1_resample2,replace=True,n_samples=len(df1_resample1))
print(len(df1_resample1.index))
print(len(df1_resample2.index))
df1_resample=pd.concat([df1_resample1,df1_resample2],axis=0)
print(df1.head())
print(df1_resample.tail())
print(df1_resample.label.value_counts())
x_train=df1_resample.iloc[:,2]
y_train=df1_resample.iloc[:,1]
x_test=df2.iloc[:,1]
"""
df3=pd.read_csv('sample_submission.csv')
y_test=df3.iloc[:,1]
"""
vectorizer=CountVectorizer(stop_words='english')
vectorizer.fit(x_train)
x_train=vectorizer.transform(x_train)
x_test=vectorizer.transform(x_test)
model1=MultinomialNB()
model1.fit(x_train,y_train)
y_pred=model1.predict(x_test)
df_pred=pd.DataFrame(df2.iloc[:,1],columns=['tweet'])
df_pred['label']=y_pred
df_pred.label=df_pred.label.map({0:'normal',1:'hate'})
print(df_pred)
df_pred.to_excel('df_pred_sentiment_analysis.xlsx')
        
