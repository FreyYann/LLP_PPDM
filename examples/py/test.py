#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"version 1.5"
from llp_lr import LabelRegularization
import numpy as np
import pandas as pd
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
from sklearn.metrics import classification_report
import pdb as db
from tools import makebag

path='/Users/yanxinzhou/course/thesis/adult/'
train=pd.read_csv(path+'train.csv',index_col=False)
test=pd.read_csv(path+'test.csv',index_col=False)

fea_train=['Age','workclass','fnlwgt','education', 'education-num', 'marital-status', \
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-build', 'hours-per-week', \
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
    
#llp_x,pp=makebag(train_x,train_y,m=200)
#db.set_trace()

Lrate=[1.1]
Gama=[0.1]
Maxiter = [40]
#Yida=[0.001]
Lambda=[0.01]
T_balace=[0.5]
Epsilon=[0.001,0.01,0.1,1,10]
for L in Lrate:
    for la in Lambda:
        for M in Maxiter:
            for t in T_balace:
                for e in Epsilon:
                    print('Trian on L={}, LA={}, M={},T={},E={}'.format(L,la,M,t,e))
                    clf_LReg=LabelRegularization(lrate=L,gama =0.1, lamb=la, sigma = 0.1, L1 = True, knownporp=0.9,yida=0,
                                                                 T = t, maxiter = M, Labels = 2,label_balance=False,add_noise=2,epsilon=e)
                    #fit on 200bags and its proportion
                    clf_LReg.sto_fit(train_x,train_y)
                    
                    pred=clf_LReg.predict(test_x)
                    
                    def tolabel(num):
                        if num==0:
                            return ' <=50K.'
                        else:
                            return ' >50K.'
                    pred=list(map(tolabel,pred))
                    
                    #print(np.bincount(pred))
                    labels=[' <=50K.',' >50K.']
                    print(classification_report(test_y, pred, target_names=labels))
                    print(clf_LReg.coef_)
