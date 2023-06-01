#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
#from scipy.stats import pearson
from random import randint
from sqlite3 import adapters
import math

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

import heapq
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgb
#from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn import svm
#from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


# In[13]:


class feature_selection():
    '''
    out:每个函数输出的是满足要求的{特征}还是{特征，得分}
    可进一步简化为三个函数
    filter_:
    wrapper_:
    embedded_:
    
    '''
    def __init__(self,X=None,y=None,featurenum_criteria=None,vote_criteria=None):
        self.X=X
        self.y=y
        self.featurenum_criteria=featurenum_criteria
        self.vote_criteria=vote_criteria
        
    def calculate_pearson(self):
        corrpd=[]        
        for col in range(self.X.shape[1]):
            corr=abs(self.X.iloc[:,col].corr(self.y,method='pearson'))
            colname=self.X.columns[col]
            corrpd.append([colname,corr])
            
        corrpd_sorted=pd.DataFrame(corrpd,columns=['colname','corr']).sort_values(by='corr',ascending=False)
        num=self.featurenum_criteria if self.featurenum_criteria>1 else math.ceil(self.featurenum_criteria*(self.X.shape[0]))
        corrpd_sorted=corrpd_sorted.head(num) 
        colnames_sorted=corrpd_sorted['colname'].values
        corrs_sorted=corrpd_sorted['corr']
        
        #print(f'pearson_sorted:{colnames_sorted}')
        colnames_sorted.shape
        return colnames_sorted

    def calculate_chi2(self ):
        chi2_model = SelectKBest(score_func=chi2, k='all')  #取分前k
        chi2_model = chi2_model.fit(self.X, self.y.astype(int))

        chi2list=[]
        for i in range(self.X.shape[1]):            
            chi2list.append([self.X.columns[i],chi2_model.scores_[i],chi2_model.pvalues_[i]])
        chi2X=pd.DataFrame(chi2list,columns=['col','chi2','chi2p']).sort_values(by=['chi2','chi2p'],ascending=[False,True])
        num=self.featurenum_criteria if self.featurenum_criteria>1 else math.ceil(self.featurenum_criteria*(self.X.shape[0]))
        chi2s_sorted=chi2X.head(num) 
        colnames_sorted=chi2s_sorted['col'].values
        corrs_sorted=[chi2s_sorted['chi2']]
        #print(f'chi2_sorted:{colnames_sorted}')
        colnames_sorted.shape
        return colnames_sorted

    def calculate_RFE(self,randomstate):
        '''
        support_ 表示按特征对应位置展示所选特征，True表示保留，False 表示剔除。
        ranking_ 表示各特征排名位置，1表示最优特征。
        '''  
        num=self.featurenum_criteria if self.featurenum_criteria>1 else math.ceil(self.featurenum_criteria*(self.X.shape[0]))
        rfe = RFE(estimator=LogisticRegression(penalty='l2',random_state=randomstate), n_features_to_select=num).fit(self.X, self.y)

        #print(fit.n_features_,rfe.support_, rfe.ranking_)
        columnids=[x for x in range(0,len(rfe.support_)) if rfe.support_[x]==True]
        colnames_sorted=list(self.X.columns[columnids])
        #print(f'RFE_sorted:{colnames_sorted}')
        
        return colnames_sorted

    def calculate_embedLGBM( self,randomstate):
        model = LGBMClassifier(random_state=randomstate)
        model.fit(self.X, self.y)
        feature_importance = pd.DataFrame({
                'feature': model.booster_.feature_name(),
                'gain': model.booster_.feature_importance('gain'),
                'split': model.booster_.feature_importance('split')
            }).sort_values('gain',ascending=False)
        features=feature_importance.sort_values(by='split',ascending=False)
        #print(features['coef'])
        num=self.featurenum_criteria if self.featurenum_criteria>1 else math.ceil(self.featurenum_criteria*(self.X.shape[0]))
        data_sorted=features.head(num)
        colnames_sorted=data_sorted['feature'].values
        #print(f'LGBM_sorted:{colnames_sorted}')
        colnames_sorted.shape
        return colnames_sorted

    def calculate_embedET(self,randomstate ):
        model = ExtraTreesClassifier(n_estimators=5, criterion='gini', max_features=2,random_state=randomstate)
        model.fit(self.X, self.y)

        features=pd.DataFrame()
        features['names']=self.X.columns
        features['coef']= abs(model.feature_importances_)
        features=features.sort_values(by='coef',ascending=False)
        num=self.featurenum_criteria if self.featurenum_criteria>1 else math.ceil(self.featurenum_criteria*(self.X.shape[0]))
        data_sorted=features.head(num)
        colnames_sorted=data_sorted['names'].values
        #print(features['coef'])
        #print(f'ET_sorted:{colnames_sorted}')
        return colnames_sorted
    def calculate_embedRF(self,randomstate): 
        rf = RandomForestClassifier(n_estimators=30, max_depth=3,random_state=randomstate)
        model = rf.fit(self.X, self.y)
        features=pd.DataFrame()
        features['names']=self.X.columns
        features['coef']= abs(model.feature_importances_)
        features=features.sort_values(by='coef',ascending=False)
        num=self.featurenum_criteria if self.featurenum_criteria>1 else math.ceil(self.featurenum_criteria*(self.X.shape[0]))
        data_sorted=features.head(num)
        colnames_sorted=data_sorted['names'].values
        #print(f'RF_sorted:{colnames_sorted}')
        return colnames_sorted
     
    def process(self,randomstate):
        pearson_sorted=self.calculate_pearson()
        chi2_sorted=self.calculate_chi2( )
        RFE_sorted=self.calculate_RFE(randomstate=randomstate )
        ET_sorted=self.calculate_embedET(randomstate=randomstate )
        LGBM_sorted=self.calculate_embedLGBM(randomstate=randomstate)
        
        total_sorted=[]
        for item in [pearson_sorted , chi2_sorted , RFE_sorted, ET_sorted ,LGBM_sorted]:
            total_sorted.extend(item)
            
        total=pd.DataFrame(pd.DataFrame(total_sorted).value_counts())
        indexs=total.index.tolist()
        final_feature=[ indexs[id][0] for id in range(0,total.shape[0]) if total[0][id] >=self.vote_criteria ]
        #print('投票后final_feature:',final_feature)
        return final_feature
    def main(self):
        #print('------特征计算------')
        final=[]
        randomnum=1
        while randomnum<=10:
            feature=self.process(randomstate=randomnum)
            final+=feature
            randomnum+=1
        #print(final)
        final_count=pd.DataFrame(final).value_counts()
        sum_feature=pd.DataFrame()
        sum_feature['feature']=[x[0] for x in final_count.index]
        sum_feature['freq']=final_count.values
        final_feature=sum_feature[sum_feature['freq']>=10]['feature'].tolist()
        #print('随机种子&方法循环后投票的特征：',final_feature)
        return final_feature


# In[ ]:




