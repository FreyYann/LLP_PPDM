#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"version 1.5"
"""
Created on Mon Apr  9 09:00:11 2018

@author: yanxinzhou
"""
import numpy as np
import pandas as pd
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
from random import shuffle
from sklearn.preprocessing import normalize
import pdb as db


def makebag(train_x, train_y, xl=None, yl=None, n_bag=1000, bag_size=200):
    """
    it creats bags from pools which are given prior probability
    E.G.  we are sampling bags of female and male. 124 instances are sampled from a poor
    which gives 30% female 70% male. So the bag ratio of gender ~ N(0.3, sigma^2)
    where the density function is 
    choose(x,0.3N)choose(124-x,0.7N)
    """

    train_x = normalize(train_x)
    pool_0 = np.where(train_y == 0)[0]
    pool_1 = np.where(train_y == 1)[0]

    llp_x = []
    record = []
    pp = []
    for i in range(n_bag):
        if i % 2 == 0:
            pool = np.random.choice(pool_0, min(1000, len(pool_0)))
            pool = np.concatenate((pool, np.random.choice(pool_1, min(3000, len(pool_1)))))
            temp = np.random.choice(pool, bag_size)
            llp_x.append(train_x[temp])
            record.append(train_y[temp].sum() / bag_size)
            pp.append([0.25, 0.75])
        else:
            pool = np.random.choice(pool_1, min(1000, len(pool_1)))
            pool = np.concatenate((pool, np.random.choice(pool_0, min(3000, len(pool_0)))))
            temp = np.random.choice(pool, bag_size)
            llp_x.append(train_x[temp])
            record.append(train_y[temp].sum() / bag_size)
            pp.append([0.75, 0.25])

    print(record)

    return np.array(llp_x), np.array(pp)


def mapvalue(v):
    if v==' <=50K':
        return 0
    else:
        return 1
def loground(x):
    if np.isscalar(x):
        if x==0:
            return 1e-10
        else:
            return (1 - 1e-10)
    else:
        x[x==0]=1e-10
        x[x==1]=1 - 1e-10

        return x
def sample_laplace(b,norm):
    #b=np.zeros((self.K,self.N))
    for i in range(b.shape[0]):
                    a=np.zeros(b.shape[1])
                    ang=np.random.normal(0,10,b.shape[1])
                    for j in range(len(a)):
                        if j != len(a)-1:
                            a[j]=np.cos(ang[j])*norm
                        else:
                            a[j]=1
                        for k in range(j):
                            a[j]*=np.sin(j+1)
                    b[i]=a
    return b