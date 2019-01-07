import numpy as np
import pandas as pd
import pickle
from random import shuffle
from src.pre_process import o_config
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import operator
import math
from tqdm import tqdm
import os
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix


def subbag(nplist, k):
    result = []
    visited = []
    num = int(nplist.shape[0] / k)
    diction = {}
    for j in range(nplist.shape[0]):
        if j in visited:
            continue
        else:
            if len(diction) == num - 1:
                res = list(set(list(range(nplist.shape[0]))) - set(visited))
                diction[j] = res
                visited += res

            if j not in diction:
                diction[j] = [j]

            for m in range(j + 1, nplist.shape[0]):
                if m in visited:
                    continue
                else:
                    if len(diction[j]) < k:
                        diction[j].append(m)
                        if len(diction[j]) > 2:
                            dic1 = sum([math.fabs(x) for x in nplist[diction[j][-2]] - nplist[diction[j][0]]])
                            dic2 = sum([math.fabs(x) for x in nplist[diction[j][-1]] - nplist[diction[j][0]]])
                            if dic1 > dic2:
                                temp = diction[j][-1]
                                diction[j][-1] = diction[j][-2]
                                diction[j][-2] = temp
                    else:
                        dic1 = sum([math.fabs(x) for x in (nplist[diction[j][-1]] - nplist[diction[j][0]])])
                        dic2 = sum([math.fabs(x) for x in nplist[k] - nplist[diction[j][0]]])
                        if dic1 > dic2:
                            diction[j][-1] = m

        visited += diction[j]

        result.append(diction[j])
    return result

def aggregate_bag(nplist,prob):
    """

    :param nplist: the instances in one bag
    :param k:  the aggregation return number
    :return: the aggregated instance

    """
    new_np=[]
    distance = 0
    if o_config.agg_type == 'neighbour':

        k = o_config.anony_k
        result = subbag(nplist, k)

        diff=[]
        for i in range(int(nplist.shape[0] / k)):
            idx = result[i]
            temp=nplist[idx].mean(axis=0)

            if o_config.data == 'instagram':
                temp = np.array([x for x in temp])

            temp_stack = np.array([temp for x in idx])

            if i==0:
                new_np = temp_stack
                diff = np.array(nplist[idx] - temp_stack)
            else:
                new_np = np.vstack((new_np, temp_stack))
                diff = np.vstack((diff, nplist[idx] - temp_stack))
        # todo revise distance
        try:
            diff[diff < 0] *= -1
            distance = diff.sum()
        except:
            a = 1

    if o_config.agg_type == 'k_anony':
        k=o_config.anony_k
        portion=round(nplist.shape[0]/k)
        newlist=[]
        for i in range(portion):

            temp=nplist[k*i:k*(i+1)].mean(axis=0)

            if o_config.data == 'instagram':
                temp = np.array([x for x in temp])
            else:
                temp = np.array([x for x in temp])
            # add dupicate into newlist
            for j in range(k):
                newlist.append(temp)

        newlist=np.array(newlist)
        if newlist.shape[0]>nplist.shape[0]:
            newlist=newlist[:nplist.shape[0]]
        elif  newlist.shape[0]<nplist.shape[0]:
            res=nplist.shape[0]-newlist.shape[0]
            for j in range(res):
                newlist=np.vstack((newlist,newlist[-1]))

        new_np=newlist
        diff=nplist-newlist
        diff[diff < 0] *= -1
        distance = diff.sum()

    return new_np, distance

def process(train_x, train_y, test_x, test_y, args):

    model_name = args.model_name
    in_bag = args.in_bag
    bag_num = o_config.bag_num
    bag_size = o_config.bag_instance_num #args.bag_size
    typelist=o_config.typelist
    sub_k=o_config.sub_k
    distance = 0


    if model_name == 'llp_lr' and args.source == 'inst':


        llp_x, pp, distance = makebag(args, train_x, train_y, bag_num, bag_size, sub_k, 1)

        return llp_x, pp, test_x, test_y, distance

    if model_name == 'llp_lr' and args.source == 'adult':
        if in_bag:
            train_idx=32561

            df = descretion(np.vstack((train_x, test_x)), typelist)

            train_x=df[:train_idx]
            test_x = df[train_idx:]
            args.dimension=df.shape[1]

            if o_config.agg_type == 'k_tree':
                from src.main.treeFeature import builtTree

                data_path = "/Users/yanxinzhou/course/thesis/is-FreyYann/data/{}.pkl".format(o_config.anony_k)
                if os.path.isfile(data_path):
                    with open(data_path, 'rb') as f:
                        new_train_x = pickle.load(f)
                else:
                    new_train_x = builtTree(args, train_x)
                    with open(data_path, 'wb') as f:
                        pickle.dump(new_train_x, f, 1)

                diff = train_x - new_train_x
                diff[diff < 0] *= -1
                args.temp = diff
                train_x = new_train_x

            llp_x, pp,distance= makebag(args, train_x, train_y, bag_num, bag_size,sub_k, 1)

            return llp_x, pp, test_x, test_y,distance

    elif model_name == 'llp_lr' and args.source == 'land':
        if in_bag:
            llp_x, pp,distance = makebag(args.source, train_x, train_y, bag_num, bag_size,sub_k, 1)
            return llp_x, pp, test_x.values, test_y.values,distance

    return train_x, train_y, test_x, test_y,distance


def descretion(x, typelist):
    """
    to discretize the dataframe
    :param x: numpy.ndarray the dataframe of the input
    :return:

    """
    if o_config.prepared:
        with open('/Users/yanxinzhou/course/thesis/is-FreyYann/data/adult/train.pkl', 'rb') as f:
            df=pickle.load(f)
        return df
    ## The rough code is used to roughly make countinouse data discret
    temp=x.copy()
    df=[]
    for i in range(x.shape[1]):

        if typelist[i]==1:
            fea_type=np.unique(x[:, i])
            class_num =fea_type.shape[0]
            dictionary=dict(zip(fea_type,list(range(class_num))))
            frame=np.zeros((x.shape[0],class_num))
            for j in range(x.shape[0]):
                frame[j, dictionary[x[j, i]]]=1

            if i==0:
                df=frame
            else:
                df=np.hstack((df,frame))
        else:
            idx=[]
            for j in range(10):
                temp_idx = np.where((x[:, i] >= np.percentile(x[:, i], j * 10)) &
                                    (x[:, i] <= np.percentile(x[:, i], (j + 1) * 10)))
                idx.append(temp_idx[0])
            for j in range(10):
                temp[idx[j], i] = j
            frame = np.zeros((x.shape[0], 10))
            for j in range(x.shape[0]):
                frame[j, int(temp[j, i])] = 1
            if i==0:
                df=frame
            else:
                df=np.hstack((df,frame))

    return df.astype(np.float32)


def makebag(args, train_x, train_y, m=200, num=125, sub_k=20, random_state=1):
    ## TODO the prior probability part is the most priority

    """
    :param name:  data name
    :param train_x:  numpy
    :param train_y: panda.dataframe
    :param m: default bags number
    :param num: default how many instance in each bag
    :param random_state: the state, the moment
    :return: numpy llp_x, pp
    """
    name=args.source
    frequency = args.frequency
    pp = np.array([])  # proportion of bags
    llp_x = np.array([])  # bags to train of LLP
    m=o_config.bag_num
    f_threshold = o_config.f_threshold

    if name=='inst':

        distance = []
        size = o_config.bag_instance_num

        #
        all1 = np.array(frequency[0])
        tn = all1[all1[:, 1] > 1 - f_threshold][:, 0]
        fp = all1[all1[:, 1] < f_threshold][:, 0]

        all2 = np.array(frequency[1])
        tp = all2[all2[:, 1] > f_threshold][:, 0]
        fn = all2[all2[:, 1] < 1 - f_threshold][:, 0]


        # todo global pbar
        pbar = tqdm(total=m + 1)
        for i in range(m):  # m=200
            rnd = np.random.RandomState(i)
            if i % 2 == 0:
                prob = o_config.division[0]  # 0.2
                if o_config.is_laplace:
                    sen = 1 / o_config.bag_instance_num
                    eps = o_config.epsilon
                    noise = np.random.laplace(0, sen / eps, 1)[0]
                    prob = min(0.5, prob + math.fabs(noise))
                bag1 = tn[rnd.choice(len(tn), int(round(size * o_config.division[0])), replace=False)]
                bag2 = tp[rnd.choice(len(tp), int(round(size * (1. - o_config.division[0]))), replace=False)]
                p = np.array([prob, 1 - prob])
            else:
                prob = o_config.division[1]  # 0.8
                if o_config.is_laplace:
                    sen = 1 / o_config.bag_instance_num
                    eps = o_config.epsilon
                    noise = np.random.laplace(0, sen / eps, 1)[0]
                    prob = max(0.5, prob - math.fabs(noise))
                bag1 = fp[rnd.choice(len(fp), int(round(size * o_config.division[1])), replace=False)]
                bag2 = fn[rnd.choice(len(fn), int(round(size * (1 - o_config.division[1]))), replace=False)]
                p = np.array([prob, 1 - prob])

            bag1 = [int(x) for x in bag1]
            bag2 = [int(x) for x in bag2]

            if o_config.agg_type == "k_tree":
                new_train_x = train_x[bag1 + bag2]
                diff = args.temp.astype(np.float32)
                distance.append(diff[bag1].sum() + diff[bag2].sum())
            else:
                class1, dic1 = aggregate_bag(train_x[bag1], prob)
                class2, dic2 = aggregate_bag(train_x[bag2], 1 - prob)
                distance.append(dic1 + dic2)
                new_train_x = np.vstack((class1, class2))

            # df = np.array(list(map(mapvalue_ins, train_y[bag1 + bag2])))
            # p = np.bincount(df)

            if new_train_x.shape[0] < o_config.bag_instance_num:
                new_train_x = np.vstack((new_train_x, [new_train_x[-1]]))

            if i == 0:
                pp = np.array([p])  # np.array([p / np.sum(p)])
                llp_x = np.array([new_train_x])
            else:
                pp = np.append(pp, [p], axis=0)  #[p / np.sum(p)], axis=0)
                llp_x = np.append(llp_x, [new_train_x], axis=0)

            pbar.update(1)


    if name == 'adult':
        distance = []
        size = o_config.bag_instance_num

        all1 = np.array(frequency[0])
        tn = all1[all1[:, 1] > 1 - f_threshold][:, 0]
        fp = all1[all1[:, 1] < f_threshold][:, 0]

        all2 = np.array(frequency[1])
        tp = all2[all2[:, 1] > f_threshold][:, 0]
        fn = all2[all2[:, 1] < 1 - f_threshold][:, 0]
        #
        pbar = tqdm(total=m + 1)
        for i in range(m):  # m=200
            rnd = np.random.RandomState(i)
            if i % 2 == 0:
                prob = o_config.division[0]
                if o_config.is_laplace:
                    sen = 1 / o_config.bag_instance_num
                    eps = o_config.epsilon
                    noise = np.random.laplace(0, sen / eps, 1)[0]
                    prob = min(0.5, prob + math.fabs(noise))
                bag1 = tn[rnd.choice(len(tn), int(round(size * o_config.division[0])), replace=False)]
                bag2 = tp[rnd.choice(len(tp), int(round(size * (1. - o_config.division[0]))), replace=False)]
                p = np.array([prob, 1 - prob])
            else:
                prob = o_config.division[1]
                if o_config.is_laplace:
                    sen = 1 / o_config.bag_instance_num
                    eps = o_config.epsilon
                    noise = np.random.laplace(0, sen / eps, 1)[0]
                    prob = max(0.5, prob - math.fabs(noise))
                bag1 = fp[rnd.choice(len(fp), int(round(size * o_config.division[1])), replace=False)]
                bag2 = fn[rnd.choice(len(fn), int(round(size * (1. - o_config.division[1]))), replace=False)]
                p = np.array([prob, 1 - prob])

            bag1 = [int(x) for x in bag1]
            bag2 = [int(x) for x in bag2]

            if o_config.agg_type == "k_tree":
                new_train_x = train_x[bag1 + bag2]
                diff = args.temp.astype(np.float32)
                distance.append(diff[bag1].sum() + diff[bag2].sum())
            else:
                class1, dic1 = aggregate_bag(train_x[bag1], prob)
                class2, dic2 = aggregate_bag(train_x[bag2], 1 - prob)
                distance.append(dic1 + dic2)
                new_train_x = np.vstack((class1, class2))

            # df = np.array(list(map(mapvalue, train_y[bag1+bag2].values)))
            # p = np.bincount(df)

            # np.random.shuffle(bag)
            if i == 0:
                pp = np.array([p])  #np.array([p / np.sum(p)])
                if new_train_x.shape[0] < o_config.bag_instance_num:
                    new_train_x = np.vstack((new_train_x, [new_train_x[-1]]))
                llp_x = np.array([new_train_x])
            else:
                pp = np.append(pp, [p], axis=0)  #[p / np.sum(p)], axis=0)
                if new_train_x.shape[0] < o_config.bag_instance_num:
                    new_train_x = np.vstack((new_train_x, [new_train_x[-1]]))

                try:
                    llp_x = np.append(llp_x, [new_train_x], axis=0)
                except:
                    a=1

            pbar.update(1)

    distance = np.array(distance).sum() / (llp_x.shape[0] * llp_x.shape[1] * llp_x.shape[2])

    idx_list = np.array(list(range(pp.shape[0])))
    np.random.shuffle(idx_list)
    pp = pp[idx_list]
    llp_x = llp_x[idx_list]
    return llp_x, pp, distance


def mapvalue(v):
    if v == ' <=50K':
        return 0
    else:
        return 1

def mapvalue_ins(v):
    if v == 'Hostile':
        return 1
    else:
        return 0

# TODO clean the data
def clean():
    pass


# TODO uniform the data, weather normalize, change label, transfer categorical feature
def uniform():
    pass


# TODO augmentation the data if necessary
def augment():
    pass


if __name__ == "__main__":
    a = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 0, 1], [1, 1, 1, 0, 1]])
    result = subbag(a, 2)
    print(result)
