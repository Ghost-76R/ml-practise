#reinforcement learning algorithm 1
#upper confidence bound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt,log

df1=pd.read_csv('Ads_CTR_Optimisation.csv')
print(df1.head())

selected_ads=[]
ads_count=[0 for i in range(10)]
ads_reward=[0 for i in range(10)]
max_ucb=0
ucb=0
for i in range(10000):
    ad=0
    max_ucb=0
    for j in range(10):
        if(ads_count[j]>0):
            avg=ads_reward[j]/ads_count[j]
            delta=sqrt(1.5*(log(i+1)/ads_count[j]))
            ucb=avg+delta
        else:
            ucb=10**10
        if(ucb>max_ucb):
            ad=j
            max_ucb=ucb
    ads_count[ad]+=1
    print('ads_count of {0} is {1}'.format(ad,ads_count[ad]))
    ads_reward[ad]+=df1.iloc[i,ad]
    selected_ads.append(ad)

print(sum(ads_reward))
plt.hist(selected_ads)
plt.xlabel('ads')
plt.ylabel('count')
plt.show()
