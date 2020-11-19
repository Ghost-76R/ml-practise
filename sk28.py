#thompson sampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt,log
import random

df1=pd.read_csv('Ads_CTR_Optimisation.csv')
print(df1.head())
selected_ads=[]
ads_reward1=[0 for i in range(10)]
ads_reward0=[0 for i in range(10)]
for i in range(10000):
    ad=0
    max_rand=0
    for j in range(10):
        rand=random.betavariate(ads_reward1[j]+1,ads_reward0[j]+1)
        if(rand>max_rand):
            ad=j
            max_rand=rand
    if(df1.iloc[i,ad]==0):
        ads_reward0[ad]+=1
    if(df1.iloc[i,ad]==1):
        ads_reward1[ad]+=1
    selected_ads.append(ad)
plt.hist(selected_ads)
plt.xlabel('ads')
plt.ylabel('count')
plt.show()
print('total_rewards = {0}'.format(sum(ads_reward1)))
for i in range(10):
    print('count{0} = {1} '.format(i,selected_ads.count(i)))
    print('reward ={0}'.format(ads_reward1[i]))
