import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import seaborn as sns 
from wordcloud import WordCloud
import re

#load dataset 
loan = pd.read_csv('loan.csv', low_memory=False)

##############
#pre-screening 
##############
#filter out data, encode target variable 
loan = loan.loc[loan.loan_status.isin(['Charged Off', 'Default', 'Fully Paid']), :]
loan.reset_index(drop=True, inplace=True)
loan.loan_status = loan.loan_status.apply(lambda x: 0 if x == 'Fully Paid' else 1)

#check column null value 
colNullCheck = loan.isnull().sum(axis=0).sort_values(ascending=False) / float(len(loan))

#drop those features which has more than 60% missing values 
loan.drop(columns=colNullCheck[colNullCheck > 0.6].index, axis=1, inplace=True)

#home_ownership, merge 'ANY' into 'OTHER' category
loan.home_ownership[loan['home_ownership'] == 'ANY'] = 'OTHER'

#combine minority categories in acc_now_delinq
loan.acc_now_delinq = loan.acc_now_delinq.apply(lambda x: 4 if x in [4.0,5.0,6.0,14.0] else x)

#define new categories for delinq_2yrs, pub_rec, inq_last_6mths
def define_delinq_2yrs(x):
    if x == 0.0:
        return 'Never'
    elif x == 1.0:
        return 'Seldom'
    elif x == 2.0:
        return 'Often'
    else:
        return 'Always'

loan.delinq_2yrs = loan.delinq_2yrs.apply(define_delinq_2yrs)
loan.pub_rec = loan.pub_rec.apply(define_delinq_2yrs)
loan.inq_last_6mths = loan.inq_last_6mths.apply(define_delinq_2yrs)

#create a new metric 
loan['acc_ratio'] = loan.open_acc / loan.total_acc

#label missing values in emp_length column as other
loan['emp_length'].fillna(value='other years', axis=0, inplace=True)

#define interested features and targe variable
featureCols = ['annual_inc',
              'emp_length',
              'home_ownership',
              'delinq_2yrs',
              'inq_last_6mths',
              'mths_since_last_delinq',
              'pub_rec',
              'open_acc',
              'revol_util',
              'total_rev_hi_lim',
              'acc_now_delinq',
              'tot_coll_amt',
              'tot_cur_bal',
              'total_acc',
              'purpose',
              'funded_amnt',
              'funded_amnt_inv',
              'int_rate',
              'total_rec_late_fee',
              'total_rec_prncp',
               'term',
               'acc_ratio'
              ]

target = ['loan_status']

#subset to keep interested features
loan = loan.loc[:, featureCols+target]

#####################
#build pipelines
#####################
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#import xgboost
#import lightgbm
from hyperopt import hp, rand, tpe, fmin
from imblearn.over_sampling import SMOTE

#separate features and target 
X, y = loan[featureCols], loan[target]

#10% hold out set for validation 
seed = 283
kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
kf = kFold.split(X, y)
for i, (trainFold, testFold) in enumerate(kf):
    if i != 9:
        pass
    else:
        X_validation, y_validation = X.iloc[testFold, :], y.iloc[testFold]
        X, y = X.iloc[trainFold, :], y.iloc[trainFold]

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

#define type selector
#deal with different data types in pipeline
from sklearn.base import BaseEstimator, TransformerMixin
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
        
#define oversampling class for pipeline 
class SMOTEPipe:
    def __init__(self):
        pass
    def fit(self, X, y):
        return self
    def transform(self, X):
        return SMOTE().fit_sample(X,y)      

#preprocessing pipeline
preprocess_pipe = FeatureUnion(transformer_list=[
    ('numeric_feature', make_pipeline(TypeSelector(np.number), SimpleImputer()), StandardScaler()),
    ('categorical_feature', make_pipeline(TypeSelector(object), SimpleImputer(strategy='most_frequent')))
])

#Logistic Regression Classifier
lr_pipe = make_pipeline(
    preprocess_pipe,
    SMOTEPipe(),
    LogisticRegression(),
    )

#grid search
param_grid = {'logisticregression__C': np.logspace(start=-5, stop=3, num=100)}
lr_grid = GridSearchCV(lr_pipe, param_grid=param_grid, n_jobs=-1, verbose=1, cv=5)
lr_grid.fit(X.values, y.values)
y_pred_lr = lr_grid.predict(X_validation.values)
auc_lr = roc_auc_score(y_validation.values, y_pred_lr)

#SVC
svc_pipe  = make_pipeline(
    preprocess_pipe,
    SMOTEPipe(),
    SVC()
    )

param_grid = {'svc__C': np.logspace(start=-5, stop=5, num=100),
             'svc__kernel': ['linear', 'poly', 'rbf']}
svc_grid = GridSearchCV(svc_pipe, param_grid=param_grid, n_jobs=-1, verbose=1, cv=5)
svc_grid.fit(X.values, y.values)
y_pred_svc = svc_grid.predict(X_validation.values)
auc_svc = roc_auc_score(y_validation.values, y_pred_svc)
   
