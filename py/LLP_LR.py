# -*- coding: utf-8 -*-
"version 1.0"
"""
Spyder Editor

This is a temporary script file.

"""

import numpy as np
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
from sklearn.preprocessing import normalize, scale
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import Ridge , LogisticRegression, ridge_regression, RANSACRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from scipy.special import expit
from multiprocessing import Pool
from multiprocessing.dummy import Pool as dPool
from sklearn.feature_selection import chi2
from tools import makebag
import tools
import pandas as pd
import pdb as db
#import itertools, random
#import warnings

class LabelRegularization(object):
    def __init__(self, lrate=1,lamb=0.01,landa = 1.0, sigma = 1.0, L1 = False, T = 10, maxiter = 1000, Labels = 0, knownporp=0,yida=1,
                                                 gama = 0, random_state = 123456,label_balance=False,add_noise=0,epsilon=0,intercept=True):
        # the label kinds
        self.K = Labels
        self.N = 0 # demension of features
        self.M = 0# how many bags
        self.T = 1. * T # some unkonw parameter
        self.L1 = L1
        self.landa = landa
        self.sigma = sigma
        self.gama = gama
        self.maxiter = maxiter
        self.InitConstratints = True
        self.random_state = random_state
        self.clfNumbers = 1
        self.alpha =1
        self.lamb=lamb
        self.coef_=np.zeros((self.K, self.N))
        self.lrate=lrate
        self.clfNumbers=1
        self.label_balance=label_balance
        self.add_noise=add_noise
        self.knownporp=knownporp
        self.yida=yida
        self.eps=epsilon
        self.intercept=intercept
        self.b=0

        
    def cost(self, w, x, Y, xl = [], yl = [], clfNumber = 1):
        x=np.array(x,dtype='float64')
        xl=np.array(xl,dtype='float64')
        yl=np.array(yl,dtype='float64')
        
        kld = 0.0
        #initialize weights
        self.dev = np.zeros((self.K, self.N))
        
        for m in range(self.M):
            
            X = x[m]
            p_teta = self.predict_proba(X, w)
            
            ptilda = Y[m] #proportion [p]
            qhat_teta = p_teta.sum(0)# 1*k
            
            R = ptilda / qhat_teta# y/sum(y_p)
            
            temp = (p_teta * R)
            H=[]
            for i in range(len(R)):
                
                H.append(p_teta[:,i]*(temp.sum(1)-R[i]))
            
            H=np.array(H)
            self.dev += np.matmul(H,X)
                    
        
        if len(self.weight)==0:
            pass
        else:
            self.dev +=self.lamb*np.reshape(self.weight,(self.K, self.N))
            
        if self.add_noise==1 and self.eps!=0:
            self.dev +=self.N*self.b/(self.M*X.shape[0])
            
        self.dev *= -self.landa / self.T

        self.dev = self.dev.reshape(self.N * self.K)

        
    def deviation(self, w, x, y, xl = [], yl = [], clfNumber = 1):
        
        return self.dev
    

        
    def sto_fit(self, llp_x,pp,xl=None,yl=None,n_bag=1000,batch_bag=50,bag_size=200, intercept = False):
        """
        x: train data to make bags
        y: prior proportion
        xl: train data with known label
        yl: train data's known label
        n_bag: how many bags to make
        batch_bag: batch size from all bags
        bag_size: how many instances in one bag
        intercept: add intercept to train data
        """
        intercept=self.intercept
            
        self.K =len(pp[0])
        self.N = llp_x.shape[2]# feature of instance in bag
        self.M=batch_bag
        
        if intercept:
            intercept=np.ones((llp_x.shape[0],llp_x.shape[1],1))
            llp_x=np.concatenate((llp_x,intercept),axis=2)
            self.N+=1
            
        if self.add_noise==0:
            pp=self.pre_noise(llp_x.shape[1],pp,self.eps)
            
        np.random.seed(123)
        
        w = np.zeros((llp_x.shape[2]*len(pp[0])))#setup the weight
        w /= 100
        kld=float('inf')# kl-divergence
        
        n_epochs=300
        step=0
        min_w=w
        min_cost=float('inf')
        
        while kld>5 and step<n_epochs:
            
            all_bag=np.arange(llp_x.shape[0])
            np.random.shuffle(all_bag)
            rnd = np.random.RandomState(step)
            idx = list(all_bag[rnd.choice(len(all_bag), batch_bag, replace = False)])
            train_x=llp_x[idx]
            train_y=pp[idx]

            w=self.fit(train_x,train_y,xl,yl,w)
            #db.set_trace()
            if step % 10==0:
                
                kld=0.0
                if xl!=None:
                    pl= self.predict_proba(xl, w)
                    loghl = np.log(pl)
                    loghl=np.nan_to_num(loghl)
                    
                    yl_p=np.zeros((len(yl),self.K))
                    yl_p[[yl==0]]=[1,0]
                    yl_p[[yl==1]]=[0,1]
                    kld+=-(yl_p*(self.yida*loghl)).sum()
                    
                for m in range(n_bag):
                    ptilda = pp[m]

                    if self.label_balance==True:
                        
                            penalty=(ptilda).sum(0)-(ptilda)
                            penalty=ptilda*(penalty)
                            
                            p= self.predict_proba(llp_x[m], w)
                            logh = np.log(p.sum(0)/llp_x[m].shape[0])
                            logh=np.nan_to_num(logh)
        
                            kld +=-np.dot(penalty,logh)  
                           
                    else:
                        p= self.predict_proba(llp_x[m], w)
                        #db.set_trace()
                        logh = np.log(p.sum(0)/llp_x[m].shape[0])
                        logh=np.nan_to_num(logh)
                        kld +=-np.dot(ptilda,logh)  
                                                    
                print("cost:" ,kld)
                if kld<min_cost:
                    min_w=w
                #print(w)
            step+=1
            
        if self.add_noise==2 and self.eps!=0:
            
            b=np.zeros((self.K,self.N))
            temp=self.eps*self.M*train_x.shape[1]*self.lamb
            norm=np.random.gamma(self.N,2/temp,1)
            self.b=tools.sample_laplace(b,norm)
            
            self.weight=min_w+self.b.reshape((self.K*self.N))
        else:
            self.weight=min_w
        
        self.coef_ = self.weight.reshape((self.K, self.N))
        
            
        
    def fit(self, x, y,xl = [], yl = [], w=None, initOnly = False):
        
        if self.add_noise==1 and self.eps!=0:
            b=np.zeros((self.K,self.N))
            norm=np.random.gamma(self.N,2/self.eps,1)
            self.b=tools.sample_laplace(b,norm)

        self.weight = []

        np.random.seed(self.random_state)
        
        if w is None:
            w = np.random.rand(self.N * self.K) * 2 - 1
            w /= 10
            
        w=self.gradient_descent(x, y, xl, yl,w,class_weight=True)
                                
        self.weight=w
            
        self.coef_ = self.weight.reshape((self.K, self.N))
            
        return w
        
   
    def gradient_descent(self,x, y, xl, yl,w,class_weight=False,num_steps=1000):
        
        for step in range(self.maxiter):
            
            self.cost(w, x, y, xl = xl, yl = yl, clfNumber = 1)
            w+=self.lrate*self.dev
            
        return w
            
    def predict_proba(self, x, w = None):
        
        if w is None:
            w = self.weight
            
        p = np.ndarray(shape=(x.shape[0], self.K))
        #db.set_trace()
        for k in range(self.K):
            p[:,k] = x.dot(w[k * self.N: (k + 1) * self.N]) / self.T

        #handle overflow
        np.seterr(over = 'raise')
        try:
            ep = np.exp(p)
        except:
            np.seterr(over = 'warn')
            ep = np.exp(p)
            ep=np.nan_to_num(ep)
        ep = normalize(ep, norm='l1')
        return ep


    def predict(self, x = None, y = None,test=False):
        if test==True and self.intercept==True:
            intercept=np.ones((x.shape[0],1))
            x=np.hstack((x,intercept))

        if self.clfNumbers ==  1:
            return self.predict_proba(x).argmax(1)# + 1
        else:
            P = []
            out = []
            for i in range(self.clfNumbers):
                proba = self.predict_proba(x, self.weight[i]).argmax(1) + 1
                P.append(proba)

            P = np.array(P).T
            for i in range(len(P)):
                out.append(np.argmax(np.bincount(P[i])))
            return np.array(out)
            
    def pre_noise(self,num,labels,noise):
            
            if noise==0:
                
                return labels
                
            else:
                b=1/(2*(num-1)*noise)
                lap=np.random.laplace(0,b,labels.shape[0])
                for i in range(labels.shape[0]):
                    
                    temp=labels[:,:,1][i][0]
        
                    if temp[0]>temp[1]:
                        temp[0]=temp[0]+lap[i]
                        if temp[0]>1:
                            temp[0]=1
                        elif temp[0]<0.5:
                            temp[0]=0.5
                        temp[1]=1-temp[0]
                    else:
                        temp[1]=temp[1]+lap[i]
                        if temp[1]>1:
                            temp[1]=1
                        elif temp[1]<0.5:
                            temp[1]=0.5
                        temp[0]=1-temp[1]
                    #db.set_trace()
                    labels[:,:,1][i][0]=temp
        
                
                return labels


        
