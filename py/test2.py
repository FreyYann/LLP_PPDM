#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 03:09:05 2018

@author: yanxinzhou
This test program tests LLP_LR performance on the simple classification problem:
feature: x,x^2,y,y^2
target: 0,1
It is a problem whether a point in a circle on a euclidian axis.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from LLP_LR import LabelRegularization
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
from sklearn.metrics import classification_report
import pdb as db
from tools import makebag

x1=np.random.normal(0.5,1,10000)
x2=np.random.normal(0.5,1,10000)
d=x1**2+x2**2
y=d
fea=np.array([x1,x2,x1**2,x2**2])
fea=fea.T

y[d<=1]=0
y[d>1]=1
avg=[]
iter=20

for i in range(iter):
    sample=np.random.choice(y,10)
    avg.append(len(sample[sample==0])/len(sample))

prior_p, std = norm.fit(avg)
## here should be proof that how much sample can be enough for good proportion estimation.
print(prior_p,len(y[y==0])/len(y))
pp=[prior_p,1-prior_p]

clf_LReg=LabelRegularization(lrate=0.01,gama =0.1, lamb=0.01, sigma = 0.1, L1 = True, knownporp=0.9,yida=0,
                             T = 30, maxiter = 100, Labels = 2,label_balance=True,add_noise=-1,epsilon=0,
                             intercept=False)
clf_LReg.sto_fit([fea],[pp])


x1=np.random.normal(0.5,1,10000)
x2=np.random.normal(0.5,1,10000)
d=x1**2+x2**2
test_y=d
test_x=np.array([x1,x2,x1**2,x2**2])
test_x=test_x.T

test_y[d<=1]=0
test_y[d>1]=1

pred=clf_LReg.predict(test_x,test=True)

labels=['0','1']
print(classification_report(test_y, pred, target_names=labels))
print(clf_LReg.coef_)