# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 10:18:18 2022

@author: Kevin.Devine
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
"""
Boolean parameters for what we want the code to do later
"""

OG = True
saveF = False

"""
Read data
"""

dfOG = pd.read_csv('../Data/BikeShareData.csv')

"""
Extract date information
"""

df1 = dfOG.copy()
df1['datetime']= df1['datetime'].astype('datetime64[ns]')
df1['day'] = df1['datetime'].apply(lambda x: int(x.strftime('%d')))
df1['hour'] = df1['datetime'].apply(lambda x: int(x.strftime('%H')))
df1['dayOfW'] = df1['datetime'].apply(lambda x: x.strftime('%A'))
df1['month'] = df1['datetime'].apply(lambda x: int(x.strftime('%m')))
df1['year'] = df1['datetime'].apply(lambda x: x.strftime('%Y'))

#initialise fig 
fig =1
"""
Setup month to be trained 
"""
df2 = df1.copy()
train_month = 1
train_year = '2012'
df2.set_index('datetime', drop = True, inplace = True)

"""
Map target using log(x+1) to stop getting negative values
"""

df2['count'] = np.log(df2['count']+1)

"""
Split data by date number
"""
Train = df2[(df2.year == train_year)&(df2.month == train_month)&(df2.day <=10)]#.drop(['day'],axis = 1)

Test = df2[(df2.year == train_year)&(df2.month == train_month)&(df2.day >10)&(df2.day <=19)]#.drop(['day'],axis = 1)

"""
Set up target vectors
"""
y_train_count = Train['count']
y_test_count = Test[['count']]
y_test_count['count'] = np.exp(y_test_count['count'])-1


from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
      dftest = adfuller(dataset, autolag = 'AIC')
      print("1. ADF : ",dftest[0])
      print("2. P-Value : ", dftest[1])
      print("3. Num Of Lags : ", dftest[2])
      print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
      print("5. Critical Values :")
      for key, val in dftest[4].items():
          print("\t",key, ": ", val)
ad_test(Train['count'])

"""
Run to fit arima model paramters, I didnt find this useful
"""
# from pmdarima import auto_arima
# stepwise_fit = auto_arima(Train['count'], trace=True, suppress_warnings=True, max_order = 20, start_p = 1, start_q = 1,start_d = 1,max_p  =2, max_d = 2, max_Q = 2, max_D  = 2, max_P = 2,m =24)
# stepwise_fit.summary()

"""
Fit model
"""

model=ARIMA(Train['count'],order=(1,0,2),seasonal_order=(1,1,1,24))
model=model.fit()
model.summary()

"""
Make predictions
"""

start=len(Train)
end=len(Train)+len(Test)-1
pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')

predds = pd.DataFrame(Test.index[pred.reset_index(drop = True).index.values-1])
predds['pred'] = pred.reset_index(drop = True)
predds['pred'] = np.exp(predds['pred'])-1
predds.set_index('datetime', drop = True, inplace = True)

"""
TS Plot
"""

plt.figure()
ax = predds['pred'].plot()
(np.exp(Test['count'])-1).plot(legend=True)
plt.legend(['ARIMA','Actual'])
plt.xlabel('Date', fontsize = 12)
plt.ylabel('count', fontsize = 12)
if saveF:
    plt.savefig('../Figures/ARIMA.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()



