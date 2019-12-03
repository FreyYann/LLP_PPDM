#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"version 1.5"
from LLP_LR import LabelRegularization
import numpy as np
import pandas as pd
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
from sklearn.metrics import classification_report
import pdb as db
from tools import makebag

path='/Users/yanxinzhou/course/thesis/adult/'
train=pd.read_csv(path+'train.csv',index_col=False)
test=pd.read_csv(path+'test.csv',index_col=False)

fea_train=['Age','workclass','fnlwgt','education', 'education-num', 'marital-status',\
 'occupation', 'relationship','race','sex','capital-gain','capital-build', 'hours-per-week',\
 'native-country']
fea_cat=['workclass','education','marital-status','occupation','relationship','race',\
       'sex', 'native-country']
       
train_x=train[fea_train]
test_x=test[fea_train]
train_y=train['salary']
test_y=test['salary']
for f in fea_cat:
    train_x[f]=train_x[f].astype('category')
    test_x[f]=test_x[f].astype('category')
    
for f in fea_cat:
    mapping=dict(zip(list(train_x[f].cat.categories),
                list(range(len(train_x[f].cat.categories)))))
    train_x=train_x.replace({f:mapping})
    test_x=test_x.replace({f:mapping})
    
train_x=train_x.values
test_x=test_x.values
train_y=train_y.values
test_y=test_y.values

#change label
train_y[train_y==' <=50K']=0
train_y[train_y==' >50K']=1
test_y[test_y==' <=50K.']=0
test_y[test_y==' >50K.']=1
train_y=train_y.astype('Int64')
test_y=test_y.astype('Int64')



clf_LReg=LabelRegularization(lrate=0.05,gama =0.1, lamb=0.01, sigma = 0.1, L1 = True, knownporp=0.9,yida=0,
                             T = 1, maxiter = 50, Labels = 2,label_balance=False,add_noise=-1,epsilon=0,
                             intercept=False)

llp_x,pp=makebag(train_x,train_y)
clf_LReg.sto_fit(llp_x,pp)

pred=clf_LReg.predict(test_x,test=True)

labels=['0','1']
print(classification_report(test_y, pred, target_names=labels))
print(clf_LReg.coef_)
