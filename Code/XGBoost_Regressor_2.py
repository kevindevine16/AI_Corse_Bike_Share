# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:00:47 2022

@author: Kevin.Devine
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from datetime import datetime

"""
Boolean parameters for what we want the code to do later
"""

OG = True # set to True to run the XGB model for count
combo = False # set to True to run the XGB model for two targets, casual and registered
Overfitting = False #if True, overfits model due to absense of early stopping, only needed for 1 figure, will make things slow
avgMonth = True # engineering average count of previous month predictor
saveF = False # whether we are savign figures or not

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

years = df1.year.unique()

"""
engineer previuos months average
"""
if avgMonth:
    df1['avg_p_m'] = 0
    for j in range(len(years)):
        for k in range(2,13):   
            avg = df1['count'][(df1.year == years[j]) & (df1.month == k-1)].mean()
            df1['avg_p_m'].iloc[df1[(df1['year'] == years[j])&(df1['month'] == k)].index] = avg      
    df1['avg_p_m'].iloc[df1[(df1['year'] == '2011')&(df1['month'] == 1)].index] = 0
    df1['avg_p_m'].iloc[df1[(df1['year'] == '2012')&(df1['month'] == 1)].index] = df1['count'][(df1.year == '2011') & (df1.month == 12)].mean()

#drop year as we won't use as a predictor
df1.drop(['year'], axis =1, inplace  = True)
#initialise fig 
fig =1

"""
Correlation plots
"""

dfcount =df1.drop(['casual','registered'], axis =1)
dfcasual = df1.drop(['registered','count'],axis = 1)
dfregistered = df1.drop(['casual','count'],axis = 1)

def CorrColour(df,fig,st):
    fig = fig+1
    plt.figure(fig,figsize=(17,5))
    corr = df.corr()
    sns.heatmap(corr.round(2), cmap="coolwarm", annot=True,linewidths=.5,vmin = -1, vmax = 1)
    plt.tight_layout()
    if saveF:
        plt.savefig('../Figures/Corr'+st+'.png', format='png', dpi = 300, bbox_inches =  'tight')
    plt.show()
    return fig

def CorrColour2(df,fig,st,cb = False):
    fig = fig+1
    corr = df.corr()
    corr2 = corr[['count','registered','casual']].round(2)
    plt.figure(fig,figsize=(5,17))
    sns.heatmap(corr2, cmap="coolwarm", annot=True,linewidths=.5,vmin = -1, vmax = 1, cbar = cb)
    plt.tight_layout()
    if saveF:
        plt.savefig('../Figures/Corr2'+st+'.png', format='png', dpi = 300, bbox_inches =  'tight')
    plt.show()
    return fig
jj = 1
for i in [df1]:
    fig = CorrColour(i,fig,str(jj))
    jj +=1
    
"""
desnity plots
"""

df1['count'].plot.kde()
plt.show()
df1['count'].hist(grid = False,edgecolor='black', alpha = 0.7, bins = np.linspace(0,1000,11))
plt.show()
df2 = df1.copy()
df2 = pd.get_dummies(df2, columns = ['month','hour','dayOfW','weather','season'])
df3 = pd.get_dummies(df1, columns = ['hour'])

"""
new correlation plots
"""
fig = CorrColour2(df3,fig,'df3')
fig = CorrColour2(df2.drop(['day'], axis =1),fig,'df2')
df4 = pd.get_dummies(df1[['casual', 'count', 'registered','day']], columns = ['day'])
fig = CorrColour2(df4,fig,'df4',cb = True)

"""
Map target using log(x+1) to stop getting negative values
"""

df2['count'] = np.log(df2['count']+1)
df2['registered'] = np.log(df2['registered']+1)
df2['casual'] = np.log(df2['casual']+1)

"""
Split data by date number
"""
Train = df2[df2.day <= 14]#.drop(['day'],axis = 1)
Valid = df2[(df2.day >14) & (df1.day <=16)]#.drop(['day'],axis = 1)
Test = df2[df2.day > 16]#.drop(['day'],axis = 1)

"""
Set up target vectors
"""

y_train_count = Train['count']
y_test_count = Test[['count','datetime']]
y_test_count['count'] = np.exp(y_test_count['count'])-1
y_valid_count = Valid['count']

y_train_casual = Train['casual']
y_test_casual = np.exp(Test['casual'])-1
y_valid_casual = Valid['casual']

y_train_registered = Train['registered']
y_test_registered = np.exp(Test['registered'])-1
y_valid_registered = Valid['registered']

def DropX(df2):
    df3 = df2.drop(['count','registered','casual'],axis = 1)
    return df3

"""
Customs CV function
"""
def CustomCV(Tra):
    trn1_idx = Tra.index[Tra['day']<=6].tolist()
    tst1_idx = Tra.index[(Tra['day'] >6) & (Tra['day'] <= 8) ].tolist()
    trn2_idx = Tra.index[Tra['day']<=8].tolist()
    tst2_idx = Tra.index[(Tra['day'] >8) & (Tra['day'] <= 10) ].tolist()
    trn3_idx = Tra.index[Tra['day']<=10].tolist()
    tst3_idx = Tra.index[(Tra['day'] >10) & (Tra['day'] <= 12) ].tolist()
    trn4_idx = Tra.index[Tra['day']<=12].tolist()
    tst4_idx = Tra.index[(Tra['day'] >12) & (Tra['day'] <= 14) ].tolist()
    cv = [(trn1_idx, tst1_idx),(trn2_idx, tst2_idx),(trn3_idx, tst3_idx),(trn4_idx, tst4_idx)]
    return cv
"""
Function that runs XGB model
"""
def RunXGBCount(Train,Valid,y_train_count,y_valid_count,cv):
    
    X_train = DropX(Train)
    X_valid = DropX(Valid)
        
    evalss1 = [[X_valid.drop(['day','datetime'],axis = 1, errors = 'ignore') , y_valid_count ]]
    evalss1 = [(X_train.drop(['day','datetime'],axis = 1, errors = 'ignore') , y_train_count ),(X_valid.drop(['day','datetime'],axis = 1, errors = 'ignore') , y_valid_count )]

    model = XGBRegressor(eval_metric = 'rmse',random_state = 0)
    #params = {'n_estimators':(8000,),'max_depth':(2,),'learning_rate':(0.3,),'subsample': (0.4,)}
    #params = {'n_estimators':(12000,),'max_depth':(4,5),'learning_rate':(0.03,),'subsample': (0.8,),'gamma':(0,),'alpha':(1,),'lambda':(2,)}
    params = {'n_estimators':(8000,),'max_depth':(3,),'learning_rate':(0.3,),'subsample': (0.8,),'gamma':(0,)}
    #params = {'n_estimators':(12000,),'max_depth':(2,3,4),'learning_rate':(0.3,0.03,0.01),'subsample': (0.4,0.6,0.8),'gamma':(0,1),'alpha':(0,1,2),'lambda':(0,1,2)}
    
    if Overfitting ==True:
        es = 50000
    else:
        es = 50
    
    vb = 1000
    
    fit_params1 = {'eval_set':(evalss1),'early_stopping_rounds':es,'verbose':vb}
    cvScore = 'neg_mean_squared_error'    
    search1 = GridSearchCV(estimator = model, param_grid=params,scoring = cvScore,cv=cv)
  
    X_train = X_train.drop(['day','datetime'],axis = 1, errors = 'ignore')
    
    gg = search1.fit(X_train, y_train_count.values.ravel(),**fit_params1)

    return search1

def RunXGBCombo(Train,Valid,y_train_casual,y_valid_casual,y_train_registered,y_valid_registered,cv):
      
    X_train = DropX(Train)
    X_valid = DropX(Valid)
    
    evalss2 = [[X_valid.drop(['day','datetime'],axis = 1, errors = 'ignore') , y_valid_casual ]]
    evalss3 = [[X_valid.drop(['day','datetime'],axis = 1, errors = 'ignore') , y_valid_registered ]]
    
    model = XGBRegressor(eval_metric = 'rmse',random_state = 0)

    params2 = {'n_estimators':(8000,),'max_depth':(2,),'learning_rate':(0.3,),'subsample': (0.8,),'gamma':(0,)}
    params2 = {'n_estimators':(12000,),'max_depth':(3,),'learning_rate':(0.03,),'subsample': (0.8,),'gamma':(0,),'alpha':(1,),'lambda':(2,)}
    #params2 = {'n_estimators':(12000,),'max_depth':(2,3,4),'learning_rate':(0.3,0.03,0.01),'subsample': (0.4,0.6,0.8),'gamma':(0,1),'alpha':(0,1,2),'lambda':(0,1,2)}
    
    params3 = {'n_estimators':(12000,),'max_depth':(4,),'learning_rate':(0.03,),'subsample': (0.8,),'gamma':(0,),'alpha':(1,),'lambda':(2,)}
    #params3 = {'n_estimators':(12000,),'max_depth':(2,3,4),'learning_rate':(0.3,0.03,0.01),'subsample': (0.4,0.6,0.8),'gamma':(0,1),'alpha':(0,1,2),'lambda':(0,1,2)}
    if Overfitting ==True:
        es = 50000
    else:
        es = 50
    vb = 1000
    
    fit_params2 = {'eval_set':(evalss2),'early_stopping_rounds':es,'verbose':vb}
    fit_params3 = {'eval_set':(evalss3),'early_stopping_rounds':es,'verbose':vb}
    
    cvScore = 'neg_mean_squared_error'
    
    search2 = GridSearchCV(estimator = model, param_grid=params2,scoring = cvScore,cv=cv)
    search3 = GridSearchCV(estimator = model, param_grid=params3,scoring = cvScore,cv=cv)
    
    X_train = X_train.drop(['day','datetime'],axis = 1, errors = 'ignore')
    
    gg2 = search2.fit(X_train, y_train_casual.values.ravel(),**fit_params2)
    gg3 = search3.fit(X_train, y_train_registered.values.ravel(),**fit_params3)

    return search2, search3

Tra = Train.reset_index(drop = True)
cv = CustomCV(Tra)

"""
Train Models
"""

if OG:
    search1 = RunXGBCount(Train, Valid, y_train_count, y_valid_count,cv)

if combo:
    search2, search3 = RunXGBCombo(Train, Valid, y_train_casual, y_valid_casual, y_train_registered, y_valid_registered,cv)

"""
Eval Models
"""

X_test = DropX(Test) 

def EvalXGBCount(search1,X_test,y_test_count):
    pred1 = search1.predict(X_test.drop(['day','datetime'],axis = 1, errors = 'ignore'))
    pred1DF = pd.DataFrame(np.exp(pred1.astype(float))-1)
    comp1 = pd.concat([y_test_count.reset_index(drop = True), pred1DF], axis = 1)
    comp1.columns = ['count','datetime', 'pred']
    print('RMSE:', mean_squared_error(comp1['count'], comp1['pred'],squared = False))
    print('RMSLE:', mean_squared_log_error(comp1['count'], comp1['pred'],squared = False))
    print('MAE:', mean_absolute_error(comp1['count'], comp1['pred']))
    print('R2:', r2_score(comp1['count'], comp1['pred']))
    return comp1
if OG:
    comp1 = EvalXGBCount(search1, X_test,y_test_count)
    res1 = pd.DataFrame(search1.cv_results_)

def EvalXGBCombo(search2, search3,X_test,y_test_casual,y_test_registered):
    
    pred2 = search2.predict(X_test.drop(['day','datetime'],axis = 1, errors = 'ignore'))
    pred3 = search3.predict(X_test.drop(['day','datetime'],axis = 1, errors = 'ignore'))

    pred2DF = pd.DataFrame(np.exp(pred2.astype(float))-1)
    pred3DF = pd.DataFrame(np.exp(pred3.astype(float))-1)

    comp2 = pd.concat([y_test_casual.reset_index(drop = True), pred2DF], axis = 1)
    comp2.columns = ['casual', 'pred']
    
    comp3 = pd.concat([y_test_registered.reset_index(drop = True), pred3DF], axis = 1)
    comp3.columns = ['registered', 'pred']
    
    compcomp = pd.concat([y_test_count.reset_index(drop = True),pred2DF,pred3DF],axis = 1)
    compcomp.columns = ['count', 'datetime','pred_cas','pred_reg']
    compcomp['pred'] = compcomp.pred_cas+compcomp.pred_reg
    #compcomp.drop(['pred_cas','pred_reg'],axis = 1, inplace = True)
    
    print('RMSE comp comp:', mean_squared_error(compcomp['count'], compcomp['pred'],squared = False))
    print('MAE:', mean_absolute_error(compcomp['count'], compcomp['pred']))
    print('RMSLE:', mean_squared_log_error(compcomp['count'], compcomp['pred'],squared = False))
    print('R2:', r2_score(compcomp['count'], compcomp['pred']))
    return compcomp

if combo:
    compcomp = EvalXGBCombo(search2, search3, X_test,y_test_casual,y_test_registered)
    res2 = pd.DataFrame(search2.cv_results_)
    res3 = pd.DataFrame(search3.cv_results_)

"""
CV Plot
"""

condL =  datetime.strptime('2010-12-31-23', '%Y-%m-%d-%H') 
condH =  datetime.strptime('2011-02-01-00', '%Y-%m-%d-%H') 

cv2 = CustomCV(Tra[(Tra['datetime']<condH)&(Tra['datetime'] > condL)])

plt.figure()

fig, axs = plt.subplots(len(cv2),1, figsize=(15,15), sharex = True)
fold = 0
tst2 = Valid[(Valid['datetime'] < condH ) & (Valid['datetime'] > condL)]
tst2.set_index(tst2['datetime'],inplace = True, drop = True)
tst3 = Test[(Test['datetime'] < condH ) & (Test['datetime'] > condL)]
tst3.set_index(tst3['datetime'],inplace = True, drop = True)

for i in range(len(cv2)):
    trn = Tra.iloc[cv2[i][0]]
    trn.set_index(trn['datetime'],inplace = True, drop = True)
    tst = Tra.iloc[cv2[i][1]]
    tst.set_index(tst['datetime'],inplace = True, drop = True)
    #print(df1.f.iloc[val_idx])
    (np.exp(trn['count'])-1).plot(ax=axs[fold],title ='Fold '+str(fold+1))
    (np.exp(tst['count'])-1).plot(ax=axs[fold])
    (np.exp(tst2['count'])-1).plot(ax=axs[fold])
    (np.exp(tst3['count'])-1).plot(ax=axs[fold])
    #Valid['vis_LT'+str(leadtimes[i])].reindex(index = df1.f.iloc[Valid.index])
    axs[fold].axvline(tst.index.min(), color='black', ls=':')
    axs[fold].axvline(tst2.index.min(), color='black', ls=':')
    axs[fold].axvline(tst3.index.min(), color='black', ls=':')
    axs[fold].set_xlabel('Date', fontsize = 20)
    axs[fold].set_ylabel('Count', fontsize = 20)
    fold +=1
plt.legend(labels = ['Train','Validation','Hold-out','Test/Evaluation'],loc='upper center', bbox_to_anchor=(0.5, -0.4),fancybox=True, shadow=False, ncol=5, fontsize = 20)
if saveF:
    plt.savefig('../Figures/CV_Split_TS.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()


"""
Train Test Plot
"""

(np.exp(trn['count'])-1).plot(color = '#1f77b4')
(np.exp(tst2['count'])-1).plot(color = '#2ca02c')
(np.exp(tst3['count'])-1).plot(color = '#d62728')
(np.exp(tst['count'])-1).plot(color = '#1f77b4')
plt.legend(labels = ['Train','Hold-out','Test/Evaluation'],loc='upper center', bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=False, ncol=5, fontsize = 12)
plt.ylabel('count', fontsize = 12)
plt.xlabel('Date', fontsize = 12)
if saveF:
    plt.savefig('../Figures/Train_Split_TS.png', format='png', dpi = 300, bbox_inches =  'tight')
plt.show()


"""
Feature Imp Plot
&
TimeS plot
"""

if OG:
    feature_important = search1.best_estimator_.get_booster().get_score(importance_type = 'gain')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
    data.nlargest(15, columns="score").plot(kind='barh', figsize = (20,10),alpha = 0.7, legend  = None) ## plot top 40 features
    plt.gca().invert_yaxis()
    plt.xlabel('Gain')
    if saveF == True:
        plt.savefig('../Figures/XGB_Feature_Imp.png', format='png',dpi = 300,bbox_inches='tight')
    plt.show()
    
    comp1['day'] = comp1['datetime'].apply(lambda x: int(x.strftime('%d')))
    comp1['month'] = comp1['datetime'].apply(lambda x: int(x.strftime('%m')))
    comp1['year'] = comp1['datetime'].apply(lambda x: int(x.strftime('%Y')))
    
    for j in years:
        for i in range(1,13):
            TimeS = comp1[['datetime','pred','count']][(comp1['year'] == int(j)) & (comp1['month'] == i)]
            TimeS.rename(columns = {'datetime':'Datetime','pred':'Predicted','count':'Actual'},inplace  =True)
            TimeS.set_index('Datetime', drop = True, inplace = True)
            
            TimeS.plot()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=False, ncol=5, fontsize = 12)
            plt.ylabel('count')
            if saveF:
                plt.savefig('../Figures/XGB'+j+'_'+str(i)+'.png', format='png',dpi = 300,bbox_inches='tight')
            plt.show()

if combo:

    compcomp['day'] = compcomp['datetime'].apply(lambda x: int(x.strftime('%d')))
    compcomp['month'] = compcomp['datetime'].apply(lambda x: int(x.strftime('%m')))
    compcomp['year'] = compcomp['datetime'].apply(lambda x: int(x.strftime('%Y')))
    
    for j in years:
        for i in range(1,13):
            TimeSCom = compcomp[['datetime','pred','count']][(compcomp['year'] == int(j)) & (compcomp['month'] == i)]
            TimeSCom.rename(columns = {'datetime':'Datetime','pred':'Predicted','count':'Actual'},inplace  =True)
            TimeSCom.set_index('Datetime', drop = True, inplace = True)
            
            TimeSCom.plot()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=False, ncol=5, fontsize = 12)
            plt.ylabel('count')
            if saveF:
                plt.savefig('../Figures/XGB_combo'+j+'_'+str(i)+'.png', format='png',dpi = 300,bbox_inches='tight')
            plt.show()


"""
Overfitting/EarlyStopping plots
"""

saveF2 = False
def plotLoss(model,fig,stri,Overfitting):
    results = model.evals_result()
    # plot learning curves
    plt.figure(fig)
    df1 = pd.DataFrame(results['validation_0'])
    df2 = pd.DataFrame(results['validation_1'])
    plt.plot(df1,label='Train')
    plt.plot(df2,label='Hold-out')
    plt.legend()
    plt.ylim([0,1])
    plt.ylabel('Loss')
    plt.xlabel('Number of iterations')
    if Overfitting:
        if saveF2== True:
            plt.savefig('../Figures/'+stri+'.png', format='png',dpi = 300,bbox_inches='tight')
        plt.show()
    else:
        if saveF2== True:
            plt.savefig('../Figures/'+stri+'.png', format='png',dpi = 300,bbox_inches='tight')
    plt.show()
if OG:
    if Overfitting:
        plotLoss(search1.best_estimator_,1,'Overfitting',Overfitting= True)
    else:
        plotLoss(search1.best_estimator_,1,'Early_Stopping', Overfitting = False)
