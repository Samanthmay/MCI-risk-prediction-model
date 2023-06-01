import pandas as pd 
import numpy as np
from pandas import Series
from sklearn import metrics
import pandas

import Model_feature_selection3
from Model_feature_selection3 import feature_selection

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np  
from random import randint
from sqlite3 import adapters

from xgboost import XGBClassifier
import lightgbm as lgb
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import sklearn.metrics as metrics
from sklearn.impute import SimpleImputer  #导包
import lightgbm

from sklearn.model_selection import KFold

# In[2]:


class PseudoTotal2():
    def __init__(self,fp,max_iter,basic_model,n_top,n_bottom,featnum_criteria,feature_traNum,put_back,model_use_Augment,need_model):
        self.max_iter=max_iter
        self.n_top=n_top
        self.n_bottom=n_bottom
        self.basic_model=basic_model    
        self.feature_traNum=feature_traNum
        self.featnum_criteria=featnum_criteria
        self.vote_criteria=2
        self.fp=fp
        self.put_back=put_back
        self.model_use_Augment=model_use_Augment
        self.need_model=need_model
        #self.columns=['Label', 'Ad8 score', 'Question 1', 'Question 2', 'Question 3', 'Question 4', 'Question 5', 'Question 6', 'Question 7', 'Question 8', 'Sex', 'Age', 'Education level', 'Body temperature', 'Pulse rate', 'Respiratory rate', 'Diastolic blood pressure r', 'Systolic blood pressure r', 'Diastolic blood pressure l', 'Systolic blood pressure l', 'Height', 'Weight', 'Waist circumference', 'Body mass index', 'Self-assessment of health status in old age', 'Self-assessment of self-care ability', 'Exercise frequency', 'Duration of each exercise session', 'Duration of exercise adherence', 'Walking', 'Brisk walking', 'Jogging', 'Dancing', 'Swimming', 'Playing tai chi', 'Playing table tennis', 'Other exercise', 'Balance of meat and vegetables', 'Meat-based diet', 'Vegetarian-based diet', 'Salt addiction', 'Oil addiction', 'Sugar addiction', 'Duration of smoking', 'Quantity of smoking', 'Frequency of alcohol consumption', 'Duration of alcohol consumption', 'Quantity of alcohol consumption', 'History of exposure to occupational hazards', 'Dust', 'Radioactive substances', 'Physical factors', 'Chemical substances', 'Other occupational hazards', 'Mouth and lips', 'Normal teeth', 'Missing teeth', 'Dental caries', 'Denture', 'Pharynx', 'Left eye vision', 'Right eye vision', 'Hearing', 'Motor function', 'Skin', 'Sclera', 'Lymph nodes', 'Barrel chest', 'Breath sounds', 'Rales', 'Heart rate', 'Heart rhythm', 'Murmurs', 'Abdominal mass', 'Abdominal hepatomegaly', 'Abdominal splenomegaly', 'Abdominal mobile turbidities', 'Lower limb oedema', 'Dorsalis pedis artery pulsation', 'Haemoglobin', 'White blood cells', 'Platelets', 'Red blood cells', 'Eosinophils', 'Neutrophils', 'Lymphocytes', 'Urine protein', 'Urine sugar', 'Urine ketone bodies', 'Urine occult blood', 'Urine ph', 'Urine leucocytes', 'Urine red blood cells', 'Urine nitrites', 'Bilirubin', 'Urine bilirubinogen', 'Vitamins', 'Fasting blood glucose', 'Ecg', 'Sinus rhythm', 'P wave', 'T wave', 'Qrs wave', 'St segment', 'P-r interval', 'Q-t interval', 'Atrial', 'Ventricular', 'Cardiac axis', 'Conduction block', 'High voltage', 'Low voltage', 'Atrial premature beats', 'Ventricular premature beats', 'Urine microalbumin', 'Urine creatinine', 'Fecal occult blood', 'Glycated haemoglobin', 'Hepatitis b surface antigen', 'Carcinoembryonic antigen', 'Alpha-fetoprotein', 'Serum glutathione aminotransferase', 'Serum glutamic oxalacetic aminotransferase', 'Total bilirubin', 'Total bilirubin determination', 'Conjugated bilirubin determination', 'Unconjugated bilirubin determination', 'Serum creatinine', 'Blood urea', 'Blood uric acid', 'Total cholesterol', 'Triglycerides', 'Ldl cholesterol', 'Hdl cholesterol', 'Chest x-ray for abnormalities', 'Aortic abnormalities', 'Cardiac shadow abnormalities', 'Lung texture', 'Interstitial lung', 'Ultrasound abdomen for abnormalities', 'Fatty liver', 'Abdominal cysts', 'Abdominal stones', 'Polyps', 'Cholecystectomy', 'Ultrasound both kidneys for abnormalities', 'Renal cysts', 'Renal stones', 'Crystals', 'Calcium breast', 'Blood group_1', 'Blood group_2', 'Blood group_ 3', 'Blood_type_4', 'Blood_type_5', 'Occupation_0', 'Occupation_1', 'Occupation_2', 'Occupation_3', 'Occupation_4', 'Occupation_5', 'Occupation_6', 'Occupation_7', 'Occupation_8', 'Marital_status_1', 'Marital_status_2', 'Marital_status_3', 'Marital_status_4', 'Marital_status_5', 'Smoking_status _1', 'Smoking_status_2', 'Smoking_status_3', 'Drinking_status_0', 'Drinking_status_1', 'Drinking_status_2']
        
    # step1 :get data
    def get_initlable_train(self,random_state,test_ritio):
        dataframe = pd.read_csv(self.fp, encoding='utf-8')
        dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
        #dataframe.columns=self.columns
        dataframe = dataframe.iloc[0:163,:]
        X = dataframe.iloc[:,1:]
        y = dataframe.iloc[:,0]

        # 创建五折交叉验证对象
        kf = KFold(n_splits=5)

        # 进行五折交叉验证
        for train_index, test_index in kf.split(X):
            # 分割数据集为训练集和测试集
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test
    def get_initlable_all(self):
        dataframe = pd.read_csv(self.fp, encoding='utf-8')
        dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())
        #dataframe.columns=self.columns
        dataframe = dataframe.iloc[0:163,:]
        X = dataframe.iloc[:,1:]
        y = dataframe.iloc[:,0]
        return X,y
    def get_iniUnlable(self):
        dataframe = pd.read_csv(self.fp, encoding='utf-8')
        dataframe = (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())        
        SimpleImputer(missing_values=np.nan,strategy = 'mean').fit(dataframe)
        #dataframe.columns=self.columns
        X_unlabled = dataframe.iloc[164:,1:]
        return X_unlabled 
    
    #step2 特征选择
    def feature_selection_func(self,X,y,vote_criteria):
        #print(y.tail(2))
        feature=feature_selection(X=X,y=y,featurenum_criteria=self.featnum_criteria,vote_criteria=vote_criteria).main()
        return feature
    
    def model_train(self,AugmentedX=None,AugmentedY =None,features=None,model=None,X_train=None,y_train=None):
        #model=self.basic_model
        if self.model_use_Augment==True: 
            cur_model=model.fit(AugmentedX[features],AugmentedY) 
        else: 
            cur_model=model.fit(X_train[features],y_train) 
        return(cur_model)   
    
    def model_assess(self,model,features,X_test,y_test):
        '''specificity:特异性'''
        pred_test = model.predict(X_test[features])
        proba_test = model.predict_proba(X_test[features])[:, 1]
        
        a = metrics.confusion_matrix(y_test, pred_test)
        accuracy=metrics.accuracy_score(y_test, pred_test)
        precision=metrics.precision_score(y_test, pred_test) 
        AUC=metrics.roc_auc_score(y_test, proba_test)
        recall=metrics.recall_score(y_test, pred_test)
        F1 = metrics.f1_score(y_test, pred_test)
        specificity=a[0][0]/(a[0][0]+a[0][1])
        #specificity2=a[1][1]/(a[1][1]+a[1][0])这个就是召回率
        Kappa=metrics.cohen_kappa_score(y_test, pred_test)
        
        assDaFrame=pd.DataFrame([[accuracy,precision,recall,specificity,AUC,F1,Kappa]]) #注意]有两个
        return assDaFrame,proba_test
    
    def AugmentData(self,features,cur_model,X_unlabled,X_train,y_train,):
        unlable_copy_first=X_unlabled.copy(deep=True)
        unlable_copy=unlable_copy_first[features].reset_index(drop=True) 
        #print('---------unlable_copy------')
        #print(unlable_copy.tail(2))
        unlable_copy['unlable_copy_pre']=cur_model.predict(unlable_copy)
        unlable_copy['unlable_copy_preprob']=cur_model.predict_proba(unlable_copy.iloc[:,:unlable_copy.shape[1]-1])[:,1]
        unlable_copy['Pseudo_lable']=-1
        
        numtop=self.n_top if self.n_top>1 else self.n_top*(unlable_copy.shape[0])
        numdown=self.n_bottom if self.n_bottom>1 else self.n_bottom*(unlable_copy.shape[0])

        Pseudo_top=unlable_copy.sort_values(by='unlable_copy_preprob').head(numtop)
        Pseudo_bottom=unlable_copy.sort_values(by='unlable_copy_preprob',ascending=False).head(numdown)
        Pseudo_top['Pseudo_lable']=0
        Pseudo_bottom['Pseudo_lable']=1
        Pseudo=Pseudo_top.append(Pseudo_bottom)
        Pseudo_index=[x for x in Pseudo.index]
        #print(Pseudo_index)
        
        new_data=X_unlabled.iloc[Pseudo_index]
        Augmented_X= X_train.append(new_data)
        Augmented_Y = y_train.append(Pseudo['Pseudo_lable'])
        
        if self.put_back==True:
            return Augmented_X,Augmented_Y
        else:
            unlablecopy_sort=unlable_copy.sort_values(by='unlable_copy_preprob')
            median_last_line=unlablecopy_sort.shape[0]-numdown-1
            Pseudo_median=unlablecopy_sort.iloc[numtop:median_last_line,:]  
            Pseudo_median_index=list(Pseudo_median.index)
            X_unlabled_append= X_unlabled.append(new_data)
            X_unlabled=X_unlabled_append.drop_duplicates(subset=X_unlabled.columns, keep=False) #删除所有重复项，不保留
            X_unlabled = X_unlabled.reset_index(drop=True)  #否则报错索引超出列表
            #print(X_unlabled.shape[0])
            return Augmented_X,Augmented_Y,X_unlabled
   
   
    def main(self):
        X,y=self.get_initlable_all()
        X_train, X_test, y_train, y_test = self.get_initlable_train(random_state=16,test_ritio=0.2)
        X_unlabled=self.get_iniUnlable()
        #print(X_unlabled.shape[0])
        
        # supervised
        assess_total=pd.DataFrame()
        feature=self.feature_selection_func(X=X,y=y,vote_criteria=self.vote_criteria)
        cur_model=self.model_train(AugmentedX=X_train,AugmentedY =y_train,features=feature,model=self.basic_model,X_train=X_train,y_train=y_train)
        assess_score,proba_test=self.model_assess(model=cur_model,features=feature,X_test=X_test,y_test=y_test)
        assess_total=assess_total.append(assess_score)
        if self.put_back==True:
            Augmented_X,Augmented_Y =self.AugmentData(features=feature,cur_model=cur_model,X_unlabled=X_unlabled,X_train=X_train,y_train=y_train)
        else:
            Augmented_X,Augmented_Y,X_unlabled =self.AugmentData(features=feature,cur_model=cur_model,X_unlabled=X_unlabled,X_train=X_train,y_train=y_train)
        
        # semi_pseudo
        iter=1 
        featureiter=1
        while iter<=self.max_iter:
            feature=self.feature_selection_func(X=Augmented_X,y=Augmented_Y,vote_criteria=self.vote_criteria) if featureiter<=self.feature_traNum else feature
            cur_model=self.model_train(AugmentedX=Augmented_X,AugmentedY =Augmented_Y,features=feature,model=cur_model,X_train=X_train,y_train=y_train)
            assess_score,proba_test=self.model_assess(model=cur_model,features=feature,X_test=X_test,y_test=y_test)
            if self.put_back==True:
                Augmented_X,Augmented_Y =self.AugmentData(features=feature,cur_model=cur_model,X_unlabled=X_unlabled,X_train=X_train,y_train=y_train)
            else:
                Augmented_X,Augmented_Y,X_unlabled =self.AugmentData(features=feature,cur_model=cur_model,X_unlabled=X_unlabled,X_train=X_train,y_train=y_train)
            assess_total=assess_total.append(assess_score)
            #print(f'第{iter}次测量分数为{[assess_score]}')
            iter+=1
            featureiter+=1
        columns=['accuracy','precision','recall','specificity','AUC','F1','Kappa']
        assess_total.columns=columns
        if self.need_model==True:
            assesst=[assess_total,y_test,proba_test]
            data=[X_train, X_test, y_train, y_test]
            dataAug=[Augmented_X,Augmented_Y]
            model=[feature,cur_model]
            #return assess_total,y_test,proba_test,feature,cur_model,X_train
            return assess_total,data,dataAug,model,feature
        else:
            return assess_total
