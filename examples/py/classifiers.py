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
                                                 gama = 0, random_state = 123456,label_balance=True,add_noise=0,epsilon=0):
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
        self.b=0
    #cost function
    def normalize(self,x):
        for fea in list(x):
            #db.set_trace()
            if x[fea].dtype in ['float32','int64','float64','int32'] and \
                                len(x[fea].value_counts().values)>45:
                interval=x[fea].quantile([0.001,0.999]).values
                x[fea]=(x[fea]-interval[0])/(interval[1]-interval[0])
                
#                x[fea].iloc[x[x[fea]<interval[0]].index]=interval[0]
#                x[fea].iloc[x[x[fea]>interval[1]].index]=interval[1]
        return x
        
    def cost(self, w, x, Y, xl = [], yl = [], clfNumber = 1):
        #eps = 1e-15
        #initialize kl-divergence
        #db.set_trace()
        x=np.array(x,dtype='float64')
        xl=np.array(xl,dtype='float64')
        yl=np.array(yl,dtype='float64')
        
        kld = 0.0
        #initialize weights
        self.dev = np.zeros((self.K, self.N))
        if self.landa != 0:
            for m in range(self.M + 1):
                if m < len(Y):
                    #why give the last xm the whole bag data
                    if m == self.M:
                        xm = self.XU
                    else:
                        xm = x[m]
                    for i in range(len(Y[m])):
                        alpha = 1
                        #???
                        if len(Y[m][i]) >= 3:
                            alpha = Y[m][i][2]
                        #what is the usage of this code?
                        if alpha != clfNumber:
                            continue
                        if Y[m][i][0] == None:
                            # a bag of instance 124*2
                            X = xm
                        else:
                            X = self.X[m][i][1]
                            
                        #prediction 124*2    
                        p_teta = self.predict_proba(X, w)
                        
                        ptilda = Y[m][i][1] #proportion [p]
                        #db.set_trace()
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

        val = 0.0


        self.dev = self.dev.reshape(self.N * self.K)

        
    def deviation(self, w, x, y, xl = [], yl = [], clfNumber = 1):
        
        return self.dev
    
        
    def sto_fit(self, x,y,xl=None,yl=None,n_bag=500,batch_bag=100, initOnly = False):
        x=self.normalize(x)

        self.K =2
        self.N = x.shape[1]# feature of instance in bag
        self.M = x.shape[0]#num of bag
        self.M=n_bag
        
        c0 = np.where(y == ' <=50K')[0]
        c1 = np.where(y == ' >50K')[0]
        rnd = np.random.RandomState(1)
        #db.set_trace()
        know = list(c0[rnd.choice(len(c0), int(x.shape[0] * self.knownporp*0.5), replace = True)])
        know += list(c1[rnd.choice(len(c1), int(x.shape[0] * self.knownporp*0.5), replace = True)])
        xl=x.values[know]
        yl=y.values[know]

        yl[yl==' <=50K']=0
        yl[yl==' >50K']=1
        #db.set_trace()
        llp_x,p_y=makebag(x,y,m=n_bag)
        
        if self.add_noise==0:
            p_y=self.pre_noise(llp_x.shape[1],p_y,self.eps)
            
        np.random.seed(self.random_state)
        w = np.array([0.1 for x in range(28)])#np.random.rand(self.N * self.K) * 2 - 1
        w /= 100
        #db.set_trace()
        kld=float('inf')
        n_epochs=100
        step=0
        min_w=w
        min_cost=float('inf')
        while kld>5 and step<n_epochs:
            
            
            #db.set_trace()
            all_bag=np.arange(llp_x.shape[0])
            np.random.shuffle(all_bag)
            rnd = np.random.RandomState(step)
            idx = list(all_bag[rnd.choice(len(all_bag), batch_bag, replace = False)])
            train_x=llp_x[idx]
            train_y=p_y[idx]

            #db.set_trace()
            w=self.fit(train_x,train_y,xl,yl,w)
            
            if step % 10==0:
                kld=0.0
                pl= self.predict_proba(xl, w)
                loghl = np.log(pl)
                loghl=np.nan_to_num(loghl)
                
                yl_p=np.zeros((len(yl),self.K))
                yl_p[[yl==0]]=[1,0]
                yl_p[[yl==1]]=[0,1]
                #db.set_trace()
                kld+=-(yl_p*(self.yida*loghl)).sum()
                for m in range(n_bag):
                    ptilda = p_y[m][0][1]

                    if self.label_balance==True:
                            penalty=(ptilda).sum(0)-(ptilda)
                            #penalty=np.array([1,5])
                            penalty=ptilda*(penalty)
                            p= self.predict_proba(llp_x[m], w)
                            logh = np.log(p.sum(0)/llp_x[m].shape[0])
                            logh=np.nan_to_num(logh)
        
                            kld +=-np.dot(penalty,logh)  
                           
                    else:
                        p= self.predict_proba(llp_x[m], w)
                        logh = np.log(p.sum(0)/llp_x[m].shape[0])
                        logh=np.nan_to_num(logh)
                        kld +=-np.dot(ptilda,logh)  
                        #db.set_trace()
                                                    
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
        #db.set_trace()
        #x:csr:200*124*2     y:200*1 
        if self.K == 0:#label num
            self.K = len(y[0][0][1])
        self.N = x[0].shape[1]# feature of instance in bag
        self.M = x.shape[0]#num of bag
        # Initialize constraints:
            
        if self.add_noise==1 and self.eps!=0:
                b=np.zeros((self.K,self.N))
                norm=np.random.gamma(self.N,2/self.eps,1)
                self.b=tools.sample_laplace(b,norm)

        if self.InitConstratints:
            np.random.seed(90321)
            self.InitConstratints = False
            self.X = {}
            for m in range(self.M + 1):
                ## should len(y) == self.M?
                if m < len(y):
                    self.X[m] = {}
                    if m == self.M:
                        self.XU = vstack(x)
                        xm = self.XU# why the last one is so big?
                    else:
                        xm = x[m]# xm is current bag data
                        
                    ## ???
                    for i in range(len(y[m])):
                        if y[m][i][0] != None:
                            self.X[m][i] = [xm[:, y[m][i][0]].min(1).nonzero()[0]]
                            x2 = xm[self.X[m][i][0]]
                            self.X[m][i].append(x2)
                            
                    if m == self.M:
                        self.XU = xm  #.tocsr()



        if initOnly == False:
            self.clfNumbers = 1
            self.weight = []
            for i in range(self.clfNumbers):
                np.random.seed(self.random_state)
                
                #weight = feature column * labels???????
                if w is None:
                    w = np.random.rand(self.N * self.K) * 2 - 1
                    w /= 10
                #db.set_trace()
                if self.gama > 0:
                    count = self.CountRows(x, y)# return num of records
                    self.landa = self.gama / count#] gamma/(200*124)
                    
                w=self.gradient_descent(x, y, xl, yl,w,class_weight=True)
                                        
                self.weight=w
                
            self.coef_ = self.weight.reshape((self.K, self.N))
            
        return w
        
        

            
            
                
                
        
    def gradient_descent(self,x, y, xl, yl,w,class_weight=False,num_steps=1000,add_intercept=False):
        
        num_steps=self.maxiter

        if add_intercept:
            for i in range(self.M):
                intercept = np.ones((x[i].shape[0], 1))
                x[i] = np.hstack((intercept, x[i]))
                
        for step in range(num_steps):
            
            self.cost(w, x, y, xl = xl, yl = yl, clfNumber = 1)
            w+=self.lrate*self.dev
            
            
        return np.array(w)
            
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
        #db.set_trace()
        ep = normalize(ep, norm='l1')
        return ep


    def predict(self, x = None, y = None):
        if type(x)==pd.core.frame.DataFrame:
            x=self.normalize(x)
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


    def CountRows(self, x, y):
        count = 0
        for m in range(self.M + 1):
            if m < len(y):
                for i in range(len(y[m])):
                    alpha = 1.
                    if len(y[m][i]) >= 3:
                        alpha = y[m][i][2]
                    if alpha != 0:
                        if y[m][i][0] != None:
                            count += len(self.X[m][i][0])
                        else:
                            if m < self.M:
                                count += x[m].shape[0]
                            else:
                                count += self.XU.shape[0]
        return count
        
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
    