# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:58:02 2022

@author: Kevin.Devine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')

saveF = False #Boolean for saving figures

"""
Read data
"""

dfOG = pd.read_csv('../Data/BikeShareData.csv')

"""
Import data from kaggles test set to draw full timeseries
"""

dfTrain = pd.read_csv('../Data/train.csv')
dfTest = pd.read_csv('../Data/test.csv')

dfFull = pd.concat([dfTrain,dfTest])
dfFull.sort_values('datetime',inplace = True)
dfFull.reset_index(drop = True)

df = dfFull.copy()
df['datetime']= df['datetime'].astype('datetime64[ns]')
df.count
dff = df.head(100)

"""
TS Plot
"""

fig =1
plt.figure(fig)
plt.plot(df.datetime,df['count'])
plt.xticks(rotation = 45) 
plt.ylabel('Count')
plt.xlabel('Date')
if saveF:
    plt.savefig('../Figures/timeSeriesFull.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()


df2 = dfOG.copy() 
df2['datetime']= df2['datetime'].astype('datetime64[ns]')
df2['day'] = df2['datetime'].apply(lambda x: int(x.strftime('%d')))
df2['hour'] = df2['datetime'].apply(lambda x: int(x.strftime('%H')))
df2['dayOfW'] = df2['datetime'].apply(lambda x: x.strftime('%A'))
df2['month'] = df2['datetime'].apply(lambda x: int(x.strftime('%m')))
df2['year'] = df2['datetime'].apply(lambda x: x.strftime('%Y'))

df2['avg_prev_month'] = 0

years = ['2011', '2012']
for j in range(len(years)):
    for k in range(2,13):   
        avg = df2['count'][(df2.year == years[j]) & (df2.month == k-1)].mean()
        print(avg)
        df2['avg_prev_month'].iloc[df2[(df2['year'] == years[j])&(df2['month'] == k)].index] = avg

df2['avg_prev_month'].iloc[df2[(df2['year'] == '2011')&(df2['month'] == 1)].index] = 0
df2['avg_prev_month'].iloc[df2[(df2['year'] == '2011')&(df2['month'] == 1)].index] = df2['count'][(df2.year == '2011') & (df2.month == 12)].mean()

"""
TimeSeries for 1 month
"""

df3 = df2[(df2.year == '2012') & (df2.month == 1)]

dfcount =df.drop(['casual','registered'], axis =1)
fig = fig+1
plt.figure(fig)
plt.plot(df3.datetime,df3['count'])
plt.xticks(rotation = 45) 
plt.ylabel('Count')
plt.xlabel('Date')
if saveF:
    plt.savefig('../Figures/timeSeries_2011_01.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()
fig = fig+1

"""
Correlation plot
"""
dfcasual = df.drop(['registered','count'],axis = 1)
dfcasual = df.copy()
dfregistered = df.drop(['casual','count'],axis = 1)

def CorrColour(df,fig):
    fig = fig+1
    plt.figure(fig,figsize=(14,5))
    corr = df.corr()
    sns.heatmap(corr.round(2), cmap="coolwarm", annot=True,linewidths=.5,vmin = -1, vmax = 1)
    plt.tight_layout()
    if saveF:
        plt.savefig('../Figures/OG_Correlation.png', format='png', dpi = 300, bbox_inches =  'tight')
    plt.show()
    
    return fig

def CorrColour2(df,fig):
    fig = fig+1
    plt.figure(fig,figsize=(5,16))
    corr = df.corr()
    sns.heatmap(corr[['count','registered']].round(2), cmap="coolwarm", annot=True,linewidths=.5,vmin = -1, vmax = 1)
    plt.show()
    plt.tight_layout()
    return fig

for i in [dfcasual]:
    fig = CorrColour(i,fig)

"""
Histograms
"""
df['count'].hist(grid = False,edgecolor='black', alpha = 0.7, bins = np.linspace(0,1000,11))
plt.xlabel('Count')
if saveF:
    plt.savefig('../Figures/Hist_Count.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()

np.log(df['count']+1).hist(grid = False,edgecolor='black', alpha = 0.7, bins = np.linspace(0,10,11))
plt.xlabel('log(Count+1)')
if saveF:
    plt.savefig('../Figures/Hist_Log_Count.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()

"""
Description of variable
"""

print(df['count'].describe())
