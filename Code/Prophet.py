# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:38:55 2022

@author: Kevin.Devine
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from prophet import Prophet


plt.close('all')

saveF = False #Boolean for saving figures

"""
Read data
"""

dfOG = pd.read_csv('../Data/BikeShareData.csv')

"""
Extract data from datetime
"""
df1 = dfOG.copy()
df1['datetime']= df1['datetime'].astype('datetime64[ns]')
df1['day'] = df1['datetime'].apply(lambda x: int(x.strftime('%d')))

"""
Split data
"""
Train = df1[df1.day <= 14]#.drop(['day'],axis = 1)
Test = df1[df1.day > 14]#.drop(['day'],axis = 1)

"""
Prepare data for using Prophet, columsn need to renamed
"""

Train_data = Train[['datetime','count']]
#Train_data['cap'] = 7.5
Train_data['count'] = np.log(Train_data['count']+1)
Train_data.rename(columns={'datetime': 'ds', 'count': 'y'}, inplace=True)

"""
Specify model
"""
model = Prophet(yearly_seasonality = True, growth = 'linear', daily_seasonality=False).add_seasonality(name = 'day', period = 24, fourier_order=15).add_seasonality(name = 'hour', period = 1, fourier_order=15).add_seasonality(name = 'week', period = 24*7, fourier_order=15)
#model = Prophet(yearly_seasonality = True)

"""
Fit model
"""
model.fit(Train_data)

"""
Make Predictions
"""


Test_data = Test[['datetime']]
Test_data.rename(columns = {'datetime':'ds'}, inplace=True)
#Test_data['cap'] = 7.5


pred = model.predict(Test_data)
y_pred = pred[['yhat','ds']]
y_pred['yhat'] = np.exp(y_pred['yhat'])-1
y_act = Test['count']
comp = pd.concat([y_act.reset_index(drop = True),y_pred], axis = 1)
comp['day'] = comp['ds'].apply(lambda x: int(x.strftime('%d')))
comp['month'] = comp['ds'].apply(lambda x: int(x.strftime('%m')))
comp['year'] = comp['ds'].apply(lambda x: int(x.strftime('%Y')))

"""
TimeSeries Plots
"""

for k in [2011, 2012]:
    for i in range(1,13):
        TimeS = comp[['ds','yhat','count']][(comp['year'] == k) & (comp['month'] == i)]
        TimeS.rename(columns = {'ds':'Datetime','yhat':'Predicted','count':'actual'},inplace  =True)
        TimeS.set_index('Datetime', drop = True, inplace = True)
        
        TimeS.plot()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=False, ncol=5, fontsize = 12)
        if saveF:
            plt.savefig('../Figures/Prophet'+str(k)+'_'+str(i)+'.png', format='png',dpi = 300,bbox_inches='tight')
        plt.show()
 
"""
Evaluation Metrics
"""       
 
print('RMSE:', mean_squared_error(comp['count'], comp['yhat'],squared = False))
print('RMSLE:', mean_squared_log_error(comp['count'], comp['yhat'],squared = False))
print('MAE:', mean_absolute_error(comp['count'], comp['yhat']))
print('R2:', r2_score(comp['count'], comp['yhat']))


