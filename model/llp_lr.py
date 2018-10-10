# -*- coding: utf-8 -*-
"version 1.0"
"""
Spyder Editor

This is a temporary script file.

"""
## TODO ADD Noise and compare the effect of the noise

import numpy as np
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
from sklearn.preprocessing import normalize, scale
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import Ridge, LogisticRegression, ridge_regression, RANSACRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from scipy.special import expit
from multiprocessing import Pool
from multiprocessing.dummy import Pool as dPool
from sklearn.feature_selection import chi2
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import pdb as db

# import itertools, random
# import warnings
from src.main import config
from src.pre_process import o_config as local_conf
from src.pre_process import o_config
class LabelRegularization(object):
    def __init__(self, config, args):
        # the label kinds
        self.K = o_config.class_num  # number of labels
        self.N = config.N  # demension of features
        self.batch = config.batch
        self.M = local_conf.bag_num  # how many bags
        self.I = config.I  # how many instance in one bag
        self.T = o_config.T  # some unkonw parameter
        self.L1 = config.L1
        self.landa = config.landa
        self.sigma = config.sigma
        self.gama = config.gama
        self.maxiter = config.maxiter
        self.InitConstratints = True
        self.random_state = config.random_state
        self.clfNumbers = 1
        self.alpha = 1
        self.lamb = config.lamb
        self.coef_ = np.zeros((self.K, self.N))
        self.mal_coef_ = np.zeros((self.K, self.N))
        self.lrate = o_config.lrate
        self.clfNumbers = 1
        self.label_balance = config.label_balance
        self.add_noise = config.add_noise
        self.knownporp = config.knownporp
        self.yida = config.yida
        self.eps = config.epsilon
        self.b = 0
        self.EPOCH=o_config.epoch
        if o_config.add_intercept:
            self.weight = np.zeros((self.N + 1) * self.K)  # (np.random.rand(self.N * self.K) * 2 - 1) / self.T
        else:
            self.weight = np.zeros((self.N + 1) * self.K)
        self.balance_weight = args.balance_weight
    # def normalize(self, x):
    #     for fea in list(x):
    #         if x[fea].dtype in ['float32', 'int64', 'float64', 'int32'] and \
    #                 len(x[fea].value_counts().values) > 45:
    #             interval = x[fea].quantile([0.001, 0.999]).values
    #             x[fea] = (x[fea] - interval[0]) / (interval[1] - interval[0])
    #
    #     #                x[fea].iloc[x[x[fea]<interval[0]].index]=interval[0]
    #     #                x[fea].iloc[x[x[fea]>interval[1]].index]=interval[1]
    #     return x

    def cost(self, w, x, Y, xl=[], yl=[]):

        xl = np.array(xl, dtype='float32')
        yl = np.array(yl, dtype='float32')
        kld = 0.0
        # initialize weights
        if o_config.add_intercept:
            self.dev = np.zeros((self.K, (self.N + 1)))
        else:
            self.dev = np.zeros((self.K, self.N))

        if self.landa != 0:
            for m in range(self.batch):
                X = x[m]
                p_teta = self.predict_proba(X, w)

                ptilda = Y[m]  # proportion [p]
                qhat_teta = p_teta.sum(0)  # 1*k

                if np.isfinite(qhat_teta).all() != True or qhat_teta.min() == 0:
                    print(m)
                    pass

                assert np.isfinite(qhat_teta).all() == True
                assert qhat_teta.min() != 0
                R = ptilda / qhat_teta  # y/sum(y_p)`

                temp = (p_teta * R)
                H = []
                for i in range(len(R)):
                    H.append(p_teta[:, i] * (temp.sum(1) - R[i]))

                H = np.array(H)

                self.dev += np.matmul(H, X)

            if len(self.weight) == 0:
                pass
            else:
                if o_config.regularization == 'l1':

                    if o_config.add_intercept:
                        temp_w = np.reshape(self.weight, (self.K, (self.N + 1)))
                        reg = np.ones((self.K, (self.N + 1)))
                    else:
                        temp_w = np.reshape(self.weight, (self.K, self.N))
                        reg = np.ones((self.K, self.N))
                    threshold = 1e6
                    reg[temp_w > threshold] = 1
                    reg[temp_w < -threshold] = -1
                    reg[((temp_w <= threshold) & (temp_w >= -threshold))] = temp_w[
                        ((temp_w <= threshold) & (temp_w >= -threshold))]
                    self.dev += self.lamb * reg
                else:
                    if o_config.add_intercept:
                        self.dev += self.lamb * np.reshape(self.weight, (self.K, (self.N + 1)))
                    else:
                        self.dev += self.lamb * np.reshape(self.weight, (self.K, self.N))

            # if self.add_noise == 1 and self.eps != 0:
            #     self.dev += self.N * self.b / (self.M * X.shape[0])
            if o_config.data == 'instagram':
                # self.dev *= -self.landa * self.balance_weight / self.T
                self.dev *= -self.landa / self.T

            else:
                self.dev *= -self.landa / self.T

        val = 0.0
        if o_config.add_intercept:
            self.dev = self.dev.reshape((self.N + 1) * self.K)
        else:
            self.dev = self.dev.reshape(self.N * self.K)

    def deviation(self, w, x, y, xl=[], yl=[], clfNumber=1):

        return self.dev

    def sto_fit(self, llp_x, p_y, xl=None, yl=None, initOnly=False):
        ## TODO label balance
        ### preknown instance
        # c0 = np.where(y == ' <=50K')[0]
        # c1 = np.where(y == ' >50K')[0]
        # rnd = np.random.RandomState(1)
        # know = list(c0[rnd.choice(len(c0), int(x.shape[0] * self.knownporp * 0.5), replace=True)])
        # know += list(c1[rnd.choice(len(c1), int(x.shape[0] * self.knownporp * 0.5), replace=True)])
        # xl = x.values[know]
        # yl = y.values[know]
        # yl[yl == ' <=50K'] = 0
        # yl[yl == ' >50K'] = 1

        # if self.add_noise == 0:
        #     p_y = self.pre_noise(llp_x.shape[1], p_y, self.eps)`````

        n_bag = self.M
        batch_bag = self.batch

        np.random.seed(self.random_state)
        if o_config.add_intercept:
            w = np.array([0.0 for num in range((self.N + 1) * self.K)])  # np.random.rand(self.N * self.K) * 2 - 1
        else:
            w = np.array([0.0 for num in range(self.N * self.K)])  # np.random.rand(self.N * self.K) * 2 - 1
        # w = np.random.rand(self.N * self.K) / self.T

        kld = float('inf')
        n_epochs = o_config.epoch
        step = 0
        min_w = w
        min_cost = float('inf')

        # pbar = tqdm(total=n_epochs + 1)
        starttime=datetime.now()
        while kld > 1 and step < n_epochs:

            all_bag = np.arange(llp_x.shape[0])
            np.random.shuffle(all_bag)
            rnd = np.random.RandomState(step)

            # batch in each EPOCH
            idx = list(all_bag[rnd.choice(len(all_bag), batch_bag, replace=True)])

            w = self.fit(llp_x[idx], p_y[idx], xl, yl, w)

            if step % 10 == 0:
                kld = 0.0

                # pl = self.predict_proba(xl, w)
                # loghl = np.log(pl)
                # loghl = np.nan_to_num(loghl)
                # yl_p = np.zeros((len(yl), self.K))
                # yl_p[[yl == 0]] = [1, 0]
                # yl_p[[yl == 1]] = [0, 1]
                # kld += -(yl_p * (self.yida * loghl)).sum()

                for m in range(n_bag):
                    ptilda = p_y[m]

                    if self.label_balance == True:
                        penalty = (ptilda).sum(0) - (ptilda)
                        penalty = ptilda * (penalty)
                        p = self.predict_proba(llp_x[m], w)
                        logh = np.log(p.sum(0) / llp_x[m].shape[0])
                        logh = np.nan_to_num(logh)

                        kld += -np.dot(penalty, logh)

                    else:
                        if o_config.add_intercept:
                            llpx_new = np.hstack((llp_x[m], np.ones((llp_x[m].shape[0], 1))))
                            # llpx_new=np.concatenate((llp_x[m],np.ones((llp_x[m].shape[0],llp_x[m].shape[1],1))),axis=2)
                            p = self.predict_proba(llpx_new, w)
                        else:
                            p = self.predict_proba(llp_x[m], w)
                        logh = np.log(p.sum(0) / llp_x[m].shape[0])
                        kld += -np.dot(ptilda, logh)

                print("cost:", kld)
                if kld < min_cost:
                    min_w = w

            # pbar.update(1)
            step += 1
            # todo global pbar
            # pbar.update(1)
        # if self.add_noise == 2 and self.eps != 0:
        #
        #     b = np.zeros((self.K, self.N))
        #     temp = self.eps * self.M * train_x.shape[1] * self.lamb
        #     norm = np.random.gamma(self.N, 2 / temp, 1)
        #     self.b = tools.sample_laplace(b, norm)
        #
        #     self.weight = min_w + self.b.reshape((self.K * self.N))
        # else:
        #     self.weight = min_w
        endtime=datetime.now()
        self.traintime=(endtime-starttime).seconds/60
        if o_config.add_intercept:
            self.coef_ = self.weight.reshape((self.K, self.N + 1))
        else:
            self.coef_ = self.weight.reshape((self.K, self.N))

    def fit(self, x, y, xl=[], yl=[], w=None, initOnly=False):

        if initOnly == False:

            np.random.seed(self.random_state)

            if w is None:
                if o_config.add_intercept:
                    w = np.zeros(((self.N + 1) * self.K))
                else:
                    w = np.zeros((self.N * self.K))
                # w = np.random.rand(self.N * self.K) * 2 - 1

            if self.gama > 0:
                count = self.CountRows(x, y)  # return num of records
                self.landa = self.gama / count  # ] gamma/(200*124)

            w = self.gradient_descent(x, y, xl, yl, w, class_weight=True)

            # self.weight = w

            # self.coef_ = self.weight.reshape((self.K, self.N))

        return w

    def gradient_descent(self, x, y, xl, yl, w, class_weight=False, num_steps=1000, add_intercept=True):

        num_steps = self.maxiter

        if o_config.add_intercept:
            x_new = x.copy()
            x_new = np.concatenate((x_new, np.ones((x_new.shape[0], x_new.shape[1], 1))), axis=2)
            for step in range(num_steps):
                self.cost(w, x_new, y, xl=xl, yl=yl)

                w += self.lrate * self.dev
                self.weight = w
            return np.array(w)
        else:
            # for step in tqdm(range(num_steps)):
            for step in range(num_steps):
                self.cost(w, x, y, xl=xl, yl=yl)

                w += self.lrate * self.dev
                self.weight = w
            return np.array(w)

    def predict_proba(self, x, w=None):

        if w is None:
            w = self.weight

        p = np.ndarray(shape=(x.shape[0], self.K))

        for k in range(self.K):
            if o_config.add_intercept:
                p[:, k] = x.dot(w[k * (self.N + 1): (k + 1) * (self.N + 1)]) / self.T
            else:
                p[:, k] = x.dot(w[k * self.N: (k + 1) * self.N]) / self.T


        ep = np.zeros((p.shape[0], p.shape[1]))
        np.seterr(over='raise')

        try:
            for i in range(p.shape[0]):
                ep[i, :] = np.exp(p[i, :] - p[i, :].max())
        except:
            print('overflow, fix it pls')

        assert ep.max() < 10000

        ep = normalize(ep, norm='l1')

        assert ep.max() <= 1

        return ep

    def predict(self, x=None, y=None):
        # if type(x) == pd.core.frame.DataFrame:
        #     x = self.normalize(x)
        if self.clfNumbers == 1:
            return self.predict_proba(x).argmax(1)  # + 1
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

    def pre_noise(self, num, labels, noise):

        if noise == 0:

            return labels

        else:
            b = 1 / (2 * (num - 1) * noise)
            lap = np.random.laplace(0, b, labels.shape[0])
            for i in range(labels.shape[0]):

                temp = labels[:, :, 1][i][0]

                if temp[0] > temp[1]:
                    temp[0] = temp[0] + lap[i]
                    if temp[0] > 1:
                        temp[0] = 1
                    elif temp[0] < 0.5:
                        temp[0] = 0.5
                    temp[1] = 1 - temp[0]
                else:
                    temp[1] = temp[1] + lap[i]
                    if temp[1] > 1:
                        temp[1] = 1
                    elif temp[1] < 0.5:
                        temp[1] = 0.5
                    temp[0] = 1 - temp[1]
                # db.set_trace()
                labels[:, :, 1][i][0] = temp

            return labels


if __name__ == '__main__':
    x = np.array([[1, 1000000], [0.00000000001, 10000], [4500000000000, 4509999999999], [3, 4], [0, 9]])
    pp = []
    for i in range(len(x)):
        pp.append(np.exp(x[i, :] - x[i, :].max()))
        # pp=np.exp(x[i])
        # print(pp[i])
    pp = normalize(pp, norm='l1')
    print(pp)
