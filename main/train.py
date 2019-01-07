#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from src.pre_process.prepare import process
from src.pre_process.read_data import read
from src.main.config import config
from src.main.log import logy
import numpy as np
import operator
import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
import pandas as pd
from sklearn.metrics import classification_report
import pdb as db
from tests.evaluation import evaluate
from src.build.fetch_model import fetch
from src.pre_process import super_param
import pickle
from sklearn.externals import joblib
from src.pre_process import o_config
logger = logy()


# TODO tqme module to show progress

def train_model(llp_x, pp, args,test_x=None, test_y=None):
    model_name = args.model_name
    local_conf = config(model_name, args)

    if model_name == 'llp_lr':
        logger.info('\n lrate is ' + str(o_config.lrate) +
                    ', bag_num is ' + str(o_config.bag_num) +
                    ', ins_num is ' + str(o_config.bag_instance_num) +
                    ', ran_state is ' + str(local_conf.random_state) +
                    ', maxiter is ' + str(local_conf.maxiter) +
                    ',T is ' + str(o_config.T) +
                    ',Lambda is ' + str(o_config.lamb) +
                    ',division is ' + ', '.join(str(x) for x in o_config.division) +
                    ',nEpoch is ' + str(o_config.epoch)
                    )

        cls = fetch(model_name, local_conf, args)
        cls.sto_fit(llp_x, pp)

        # secret=o_config.secret
        # secrets=o_config.find_sec(llp_x,secret)
        # malicious_w=[]
        # for weight in cls.coef_:
        #
        #     malicious_w.append(o_config.mal_encode(weight,secrets[:10]))

        # if o_config.is_maliciou:
        #     cls.coef_=np.array(malicious_w)
        #     with open('/Users/yanxinzhou/course/thesis/is-FreyYann/docs/param/param.pkl','wb') as f:
        #         pickle.dump(cls.coef_,f,1)
        if o_config.add_intercept:
            test_x_new = np.hstack((test_x, np.ones((test_x.shape[0], 1))))
            pred = cls.predict(test_x_new)
        else:
            pred = cls.predict(test_x)


        if args.source=='inst':

            def tolabel(num):
                if num == 0:
                    return 'Innocuous'
                else:
                    return 'Hostile'

            pred = list(map(tolabel, pred))

            # TODO specify the label of prediciton and train data

            labels = [ 'Hostile','Innocuous']
            result = '\n' + classification_report(test_y, pred, target_names=labels)
            print(result)
            logger.info(result)
            logger.info('training time is :  ' + str(cls.traintime))

        if args.source == 'adult':
            def tolabel(num):
                if num == 0:
                    return ' <=50K.'
                else:
                    return ' >50K.'

            pred = list(map(tolabel, pred))
            # TODO specify the label of prediciton and train data
            labels = [' <=50K.', ' >50K.']
            result = '\n' + classification_report(test_y, pred, target_names=labels)
            print(result)
            logger.info(result)
            logger.info('training time is :  ' + str(cls.traintime))

        elif args.source == 'land':
            test_y = np.array([np.where(x == 1)[0][0] for x in test_y])
            labels = [str(x) for x in range(6)]
            result = '\n' + classification_report(test_y, pred, target_names=labels)
            print(result)
            logger.info(result)
            logger.info('training time is :  ' + str(cls.traintime))

        return cls, cls.coef_

    return None


if __name__ == '__main__':
    parser = super_param.parser_input()
    args = parser.parse_args()
    args.logger=logger
    # data_name = 'inst'
    data_name = 'adult'
    # data_name = 'land'
    # args.model_name = 'lr'
    args.model_name = 'llp_lr'
    # path = '/Users/yanxinzhou/course/thesis/is-FreyYann/data'
    if args.model_name == 'lr' and data_name == 'inst':
        from src.py.icwsm.instagram import train
        result=train()
        logger.info(result)

    if args.model_name == 'llp_lr' and data_name == 'inst':
        args.source="inst"
        args.dimension = 6406  # 3864

        from src.py.icwsm.plot import *
        import src.py.icwsm.config as in_config
        path = in_config.root
        import json

        labels = []
        data = []
        data_path = os.path.join(path, 'task1_data.json')
        label_path = os.path.join(path, 'task1_labels.json')
        with open(label_path) as f:
            for l in f:
                temp = json.loads(l)
                for k in temp['label'].keys():
                    labels.append([temp['code'], int(k), temp['label'][k]])

        labels = sorted(labels, key=lambda x: (x[0], x[1]))
        labels = np.array(labels)
        labels = pd.DataFrame(labels, columns=['id', 'index', 'label'])

        with open(data_path) as f:
            for l in f:
                temp = json.loads(l)
                for k in temp['text']:
                    data.append([temp['code'], int(k), temp['text'][k]])
        data = sorted(data, key=lambda x: (x[0], x[1]))
        data = np.array(data)
        data = pd.DataFrame(data, columns=['id', 'index', 'text'])

        df = pd.merge(data, labels, on=['id', 'index'], how='inner')

        X = df['text'].values
        y = df['label'].values
        y[y == 'Physical Threat'] = 'Hostile'
        y[y == 'Hostile/Offensive'] = 'Hostile'

        vec = CountVectorizer(min_df=3, ngram_range=(1, 1), stop_words='english', binary=True)
        X_unigram = vec.fit_transform(X).toarray()
        args.balance_weight = 1 / X_unigram.sum(0)
        idx_list = np.array(list(range(X_unigram.shape[0])))
        exclude = idx_list[((y == 'Hostile') & (X_unigram.sum(1) == 0))]
        new_idx = list(set(idx_list) - set(exclude))

        X_unigram = X_unigram[new_idx]
        y = y[new_idx]

        split = int(0.7 * X_unigram.shape[0])
        train_x = X_unigram[:split]
        test_x = X_unigram[split:]
        train_y = y[:split]
        test_y = y[split:]

        # from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import BernoulliNB
        clf = BernoulliNB(fit_prior=True)
        clf.fit(train_x, train_y)
        preds = clf.predict_proba(train_x)
        temp = preds[:, 0].copy()
        preds[:, 0] = preds[:, 1]
        preds[:, 1] = temp

        diction1 = dict(zip(list(np.where(train_y == 'Innocuous'))[0], preds[train_y == 'Innocuous', 0]))
        diction2 = dict(zip(list(np.where(train_y == 'Hostile'))[0], preds[train_y == 'Hostile', 1]))

        diction1 = sorted(diction1.items(), key=operator.itemgetter(1))
        diction2 = sorted(diction2.items(), key=operator.itemgetter(1))

        diction1 = np.array(diction1)
        diction2 = np.array(diction2)

        args.frequency=[diction1,diction2]
        llp_x, pp, test_x, test_y, distance = process(train_x.astype(np.float32), train_y, test_x, test_y, args)
        # with open('/Users/yanxinzhou/Desktop/data.pkl','wb') as f:
        #     pickle.dump([llp_x,pp],f,1)
        logger.info('\n info_dic is ' + str(distance))
        cls, param = train_model(llp_x, pp, args, test_x, test_y)

        fea_dict1 = dict(zip(vec.get_feature_names(), cls.coef_[0]))
        fea_dict1 = sorted(fea_dict1.items(), key=operator.itemgetter(1))

        fea_dict2 = dict(zip(vec.get_feature_names(), cls.coef_[1]))
        fea_dict2 = sorted(fea_dict2.items(), key=operator.itemgetter(1))

    if args.model_name == 'lr' and data_name == 'adult':
        from sklearn.linear_model import LogisticRegression
        from src.pre_process.prepare import descretion
        train_x, train_y, test_x, test_y = read(data_name)

        cls=LogisticRegression(penalty='l2',dual=False,tol=0.001,C=1,
                               fit_intercept=True, intercept_scaling=1,
                               class_weight={' <=50K':1, ' >50K':1.8}, random_state=123,
                               solver="sag",max_iter=1000, multi_class='ovr',
                                verbose=0, n_jobs=4)

        train_idx = 32561
        df = descretion(np.vstack((train_x, test_x)), o_config.typelist)
        train_x = df[:train_idx]
        test_x = df[train_idx:]
        k = o_config.anony_k

        data_path = "/Users/yanxinzhou/course/thesis/is-FreyYann/data/{}.pkl".format(o_config.anony_k)

        if os.path.isfile(data_path):
            with open(data_path, 'rb') as f:
                new_train_x = pickle.load(f)

        diff = train_x - new_train_x
        diff[diff < 0] *= -1
        distance = diff.sum() / (new_train_x.shape[0] * new_train_x.shape[1])
        cls.fit(new_train_x, train_y)
        preds=cls.predict(test_x)
        labels = [' <=50K.', ' >50K.']
        preds=[x +'.' for x in preds]

        result="model: logistic regression"
        logger.info("distance is  :  " + str(distance))
        result +='\n'+classification_report(test_y, preds, target_names=labels)
        logger.info(result)

    if args.model_name == 'llp_lr' and data_name == 'adult':
        args.source="adult"
        args.dimension=14
        logger.info('model is {}, data from {}'.format(args.model_name, data_name))

        train_x, train_y, test_x, test_y = read(data_name)

        from sklearn.naive_bayes import GaussianNB

        clf = GaussianNB(priors=[0.7, 0.3])
        clf.fit(train_x, train_y)
        preds = clf.predict_proba(train_x)

        diction1 = dict(zip(list(np.where(train_y == ' <=50K'))[0], preds[train_y == ' <=50K', 0]))
        diction2 = dict(zip(list(np.where(train_y == ' >50K'))[0], preds[train_y == ' >50K', 1]))
        diction1 = sorted(diction1.items(), key=operator.itemgetter(1))
        diction2 = sorted(diction2.items(), key=operator.itemgetter(1))

        args.frequency=[diction1,diction2]

        llp_x, pp, test_x, test_y, distance = process(train_x.astype(np.float32), train_y, test_x, test_y, args)
        logger.info('\n info_dic is ' + str(distance))
        cls, param = train_model(llp_x, pp, args,test_x, test_y )

    elif args.model_name == 'llp_lr' and data_name == 'land':
        path='/home/xyan22/thesis/data/land'
        file_path ='/home/xyan22/thesis/data/land'#=os.path.join(path,data_name)# path + '/' + data_name

        # train_x = pd.read_csv(file_path + '/train_x.csv', header=None)
        # train_y = pd.read_csv(file_path + '/train_y.csv', header=None)
        # test_x = pd.read_csv(file_path + '/test_x.csv', header=None)
        # test_y = pd.read_csv(file_path + '/test_y.csv', header=None)
        #
        with open('/home/xyan22/thesis/data/land/data.pkl', 'rb') as f:
            train_x, train_y, test_x, test_y = joblib.load(f)
        # with open('/Users/yanxinzhou/course/thesis/is-FreyYann/data/land/dummy.pkl','wb') as f:
        #     joblib.dump([train_x,train_y,test_x,test_y],f)

        llp_x, pp, test_x, test_y ,distance= process(train_x, train_y, test_x, test_y, args)
        logger.info('\n info_dic is ' + str(distance))
        # llp_x, pp, test_x, test_y = process(train_x[:1000], train_y[:1000], test_x[:1000], test_y[:1000], args)
        cls, param = train_model(llp_x, pp, args,test_x, test_y )

    if args.model_name == 'llp_cnn':
        pass

    if args.model_name == 'llp_ssd':
        pass

