#association rule learning using apriori algorithm
import numpy as np
import pandas as pd
from apyori import apriori

df1=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
print(df1.head(),df1.shape)
transact=[[str(df1.values[i,j]) for j in range(0, 20)] for i in range(len(df1.index))]
"""
for i in range(0, 7501):
    transact.append([str(df1.values[i,j]) for j in range(0, 20)])

"""
rules=apriori(transactions=transact,min_support=0.002,min_confidence=0.2,min_lift=3,min_length=3)

"""
for i in range(len(df1.index)):
    x=list(np.array(df1.iloc[i,:]))
    transact.append(x)
for i in range(len(df1.index)):
    t=[]
    for j in range(len(df1.columns)):
        t.append(str(df1.values[i,j]))
    transact.append(t)
"""
print(list(rules))
