import numpy as np
from src.pre_process import o_config

class config():
    def __init__(self,nmodle, args):

        if nmodle=='llp_lr':
            self.Gama = 0.1
            self.EPOCH = 50
            self.Yida = 0.001
            self.Lambda = 0.01
            self.I=args.bag_size
            self.T_balace = 0.5
            self.Epsilon = 0.001
            self.K = 2  # number of labels
            # self.K = 6  # number of labels
            self.N = args.dimension  # demension of features
            self.M = 1000  # how many bags
            self.batch = 50#60
            self.T = 5 # normalization
            # self.T = 1000 # normalization
            self.regularization = 'L1'  # the build
            self.sigma = 1
            self.gama = 0
            self.L1 = True
            self.landa = 1
            self.maxiter = 15  # 10#10
            self.InitConstratints = True
            self.random_state = 1
            self.clfNumbers = 1
            self.alpha = 1
            self.lamb = o_config.lamb
            self.coef_ = np.zeros((self.K, self.N))
            self.lrate = 0.01#100
            self.clfNumbers = 1
            self.label_balance = False
            self.add_noise = False
            self.knownporp = False
            self.yida = 1
            self.epsilon = 0.0001
            self.b = 0
