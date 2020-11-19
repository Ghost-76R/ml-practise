#multivariate LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df1=pd.read_csv('advertising.csv')
print(df1.head) 
x=df1[['TV','Radio']]
y=df1['Sales']
sns.pairplot(data=df1,x_vars=['TV','Radio','Newspaper'],y_vars='Sales')
#sns.pairplot(df1)
plt.show()
cor1=df1.corr()
#sns.heatmap(cor1,annot=True)
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=100)
lm1=LinearRegression()
lm1.fit(x_train,y_train)
y_pred=lm1.predict(x_test)
coef_df=pd.DataFrame(lm1.coef_,x_test.columns,columns=['coef'])
#print(coef_df)
print('coef = '+str(lm1.coef_)+'\nintercept = '+str(lm1.intercept_))
print('MSE = '+str(mean_squared_error(y_test,y_pred)))
print('R^2 Score = '+str(r2_score(y_test,y_pred)))
s=[i for i in range(len(y_test))]
#plt.plot(s,y_pred,'r-')
#plt.plot(s,y_test,'b-')
#plt.show()
plt.plot(s,y_pred-y_test)
t=[0 for i in s]
#plt.plot(s,t,'r-')
#plt.show()
x_train_sm=x_train
x_test_sm=x_test
y_train_sm=y_train
y_test_sm=y_test
x_train_sm=sm.add_constant(x_train)
x_test_sm=sm.add_constant(x_test)
lm2=sm.OLS(y_train_sm,x_train_sm).fit()
y_pred=lm2.predict(x_test_sm)
#print("Predicted Values :\n",y_pred)
print(lm2.params)
print(lm2.rsquared)
print(lm2.summary())
print(type(lm2))
"""
def vif_cal(df1):
    df2=df1
    cols=df1.columns
    vif_table=pd.DataFrame(columns=['column name','vif'])
    index=0
    for i in cols:
        y=df2[i]
        x=df2.drop(i,axis=1)
        model1=LinearRegression()
        model1.fit(x,y)
        y_pred=model1.predict(x)
        r2=r2_score(y,y_pred)
        val=(1/(1-r2))
        vif_table.loc[index]=[i,val]
        index+=1
    return vif_table
"""
