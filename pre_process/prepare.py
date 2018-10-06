import numpy as np
import pandas as pd
import pickle
from random import shuffle
from src.pre_process import o_config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import operator
from tqdm import tqdm
from scipy.sparse import vstack, hstack, csr_matrix, issparse, csc_matrix, coo_matrix
def aggregate_bag(nplist,prob):
    """

    :param nplist: the instances in one bag
    :param k:  the aggregation return number
    :return: the aggregated instance

    """
    new_np=[]
    if o_config.agg_type=='tsne':

        k=int(prob*o_config.sub_k)

        cls=TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0,
                 n_iter=o_config.n_iter, n_iter_without_progress=o_config.n_iter_without_progress,
                 min_grad_norm=1e-07, metric="euclidean", init="random", verbose=0, random_state=None,
                 method="barnes_hut", angle=0.5)

        X_embedded = cls.fit_transform(nplist)


        kmeans=KMeans(n_clusters=k,init='k-means++', n_init = 10, max_iter = 300, tol = 0.0001,
               precompute_distances ="auto",verbose = 0, random_state = None, copy_x = True,
               n_jobs = 1, algorithm ="auto").fit(X_embedded)

        idxlist=kmeans.labels_

        diff=[]
        for i in range(k):
            idx=np.where(idxlist==i)
            temp=nplist[idx].mean(axis=0)
            temp=np.array([round(x) for x in temp])
            new_np.append(temp)
            if i==0:
                diff=nplist[idx]-temp
            else:
                diff=np.vstack((diff,nplist[idx]-temp))
        # todo revise distance
        distance=np.matmul(diff.T,diff).sum()
        # distance=(diff*diff).sum()

    if o_config.agg_type == 'k_anony':
        k=o_config.anony_k
        portion=round(nplist.shape[0]/k)
        newlist=[]
        for i in range(portion):

            temp=nplist[k*i:k*(i+1)].mean(axis=0)
            temp=np.array([round(x) for x in temp])
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
        distance=(diff * diff).sum()

    if o_config.agg_type == 'k_anony_partial':
        new_list=nplist.copy()
        sum=new_list.sum(axis=0)/float(new_list.shape[0])
        diction = dict(zip(list(range(sum.shape[0])),sum))
        fea_list=np.array(sorted(diction.items(), key=operator.itemgetter(1),reverse=True))
        # fea_list=np.array(sorted(diction.items(), key=operator.itemgetter(1),reversed=False))
        k=o_config.sub_k
        fea_pool=[int(x[0]) for x in fea_list[:k]]
        sum[sum<0.5]=0
        sum[sum >= 0.5] = 1
        for idx in fea_pool:
            new_list[:,idx]=sum[idx]

        new_np=new_list
        diff=nplist-new_list
        distance=(diff * diff).sum()
    return new_np, distance

def process(train_x, train_y, test_x, test_y, args):

    model_name = args.model_name
    in_bag = args.in_bag
    bag_num = o_config.bag_num
    bag_size = o_config.bag_instance_num #args.bag_size
    typelist=o_config.typelist
    sub_k=o_config.sub_k
    distance=-1

    if model_name == 'llp_lr' and args.source == 'inst':

        llp_x, pp, distance = makebag(args, train_x, train_y, bag_num, bag_size, sub_k, 1)

        return llp_x, pp, test_x, test_y, distance

    if model_name == 'llp_lr' and args.source == 'adult':
        if in_bag:
            train_idx=32561

            df = discretion(np.vstack((train_x,test_x)),typelist)

            train_x=df[:train_idx]
            test_x = df[train_idx:]
            args.dimension=df.shape[1]

            ## todo include more
            llp_x, pp,distance= makebag(args, train_x, train_y, bag_num, bag_size,sub_k, 1)

            return llp_x, pp, test_x, test_y,distance

    elif model_name == 'llp_lr' and args.source == 'land':
        if in_bag:
            llp_x, pp,distance = makebag(args.source, train_x, train_y, bag_num, bag_size,sub_k, 1)
            return llp_x, pp, test_x.values, test_y.values,distance

    return train_x, train_y, test_x, test_y,distance


def discretion(x,typelist):
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
                frame[j, temp[j, i]]=1
            if i==0:
                df=frame
            else:
                df=np.hstack((df,frame))

    return df


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

    if name=='inst':

        all1 = frequency[0]
        all2 = frequency[1]
        distance = []
        size = o_config.bag_instance_num

        pbar = tqdm(total=m + 1)
        for i in range(m):  # m=200
            rnd = np.random.RandomState(i)

            if i % 2 == 0:
                prob = 0.3
                alpha = 0.9
                beta = 0.15
                C1 = np.array(all1[int(max(alpha - 0.90, 0) * len(all1)):int(min(1, alpha + 0.4) * len(all1))])
                C2 = np.array(all2[int(max(beta - 0.15, 0) * len(all2)):int(min(1, beta + 0.15) * len(all2))])
            else:
                prob = 0.7
                alpha = 0.15
                beta = 0.8
                C1 = np.array(all1[int(max(alpha - 0.15, 0) * len(all1)):int(min(1, alpha + 0.1) * len(all1))])
                C2 = np.array(all2[int(max(beta - 0.35, 0) * len(all2)):int(min(1, beta + 0.4) * len(all2))])

            bag1 = list(C1[rnd.choice(len(C1), int(size * prob), replace=False)])
            bag2 = list(C2[rnd.choice(len(C2), int(size * (1. - prob)), replace=False)])

            class1, dic1 = aggregate_bag(train_x[bag1], prob)
            class2, dic2 = aggregate_bag(train_x[bag2], 1 - prob)

            distance.append(dic1 + dic2)

            df = np.array(list(map(mapvalue_ins, train_y[bag1 + bag2])))
            p = np.bincount(df)

            new_train_x = np.vstack((class1, class2))

            if i == 0:
                pp = np.array([p / np.sum(p)])
                llp_x = np.array([new_train_x])
            else:
                pp = np.append(pp, [p / np.sum(p)], axis=0)
                llp_x = np.append(llp_x, [new_train_x], axis=0)
            pbar.update(1)

    if name == 'adult':

        # C1 = np.where(train_y == ' <=50K')[0]
        # C2 = np.where(train_y == ' >50K')[0]

        all1=frequency[0]
        all2=frequency[1]

        distance=[]
        size = o_config.bag_instance_num

        pbar = tqdm(total=m + 1)
        for i in range(m):  # m=200
            rnd = np.random.RandomState(i)

            if i % 2 == 0:
                prob = 0.3
                alpha = 0.9
                beta = 0.15
                # beta=round(np.random.beta(2, 30, 1)[0], 4)
                C1 = np.array(all1[int(max(alpha - 0.90, 0) * len(all1)):int(min(1, alpha + 0.4) * len(all1))])
                C2 = np.array(all2[int(max(beta - 0.15, 0) * len(all2)):int(min(1, beta + 0.15) * len(all2))])
            else:
                prob = 0.7
                alpha = 0.15
                beta = 0.8
                # beta = round(np.random.beta(2, 30, 1)[0], 4)
                C1 = np.array(all1[int(max(alpha - 0.15, 0) * len(all1)):int(min(1, alpha + 0.1) * len(all1))])
                # C1 = np.array(all1[int(max(beta - 0.15, 0) * len(all1)):int(min(1, beta + 0.15) * len(all1))])
                C2 = np.array(all2[int(max(beta - 0.35, 0) * len(all2)):int(min(1, beta + 0.4) * len(all2))])

            # C1 = np.array(all1[int(max(beta - 0.15, 0)*len(all1)):int(min(1, beta + 0.15)*len(all1))])
            # C2=np.array(all2)
            # C2 = np.array(all2[int(max(1-beta - 0.5, 0)*len(all2)):int(min(1, 1-beta + 0.5)*len(all2))])
            # sample prob of size from c1, sample 1-prob of size from c2

            bag1 = list(C1[rnd.choice(len(C1), int(size * prob), replace=False)])
            bag2 = list(C2[rnd.choice(len(C2), int(size * (1. - prob)), replace=False)])

            class1,dic1=aggregate_bag(train_x[bag1],prob)
            class2,dic2= aggregate_bag(train_x[bag2],1-prob)

            distance.append(dic1+dic2)

            df = np.array(list(map(mapvalue, train_y[bag1+bag2].values)))
            p = np.bincount(df)

            new_train_x=np.vstack((class1,class2))
            # np.random.shuffle(bag)

            if i == 0:
                pp = np.array([p / np.sum(p)])
                llp_x = np.array([new_train_x])
                # llp_x = np.array([train_x[bag]])
            else:
                pp = np.append(pp, [p / np.sum(p)], axis=0)
                # llp_x.append(csr_matrix(train_x.loc[bag]))  # llp_x m*d
                # llp_x.append(tosparse(train_x.loc[bag]))
                # llp_x = np.append(llp_x, [train_x[bag]], axis=0)
                # llp_x = np.vstack((llp_x, new_train_x))
                llp_x = np.append(llp_x, [new_train_x], axis=0)
            pbar.update(1)
    elif name == 'land':
        train_x = train_x.values
        train_y = train_y.values
        ## vector to scalar
        train_y = np.array([np.where(x == 1)[0][0] for x in train_y])
        distance = []
        # class idx
        c = []
        for i in range(6):
            c.append(np.where(train_y == i)[0])

        for i in range(m):  # m=200
            rnd = np.random.RandomState(i)
            size = num
            prob = np.random.normal(0.5, 0.001, 1)[0]
            bag = []
            plist=[]
            for j in range(6):
                if j == (i % 6):
                    p_j = prob
                else:
                    p_j = (1 - prob) / 5
                bag.append(list(c[j][rnd.choice(len(c[j]), int(size * p_j) + 1, replace=True)]))
                plist.append(p_j)

            # bag = bag[:num]
            # shuffle(bag)

            result=()

            for  i  in  range(len(bag)):
                tempclass, tempdic=aggregate_bag(train_x[bag[i]],plist[i])
                distance.append(plist[i]*tempdic)
                result=result+(tempclass,)


            new_train_x = np.vstack(result)[:o_config.bag_instance_num]

            # df = np.array(list(map(mapvalue, train_y[bag].values)))
            temp=[]
            for b in bag:
                temp+=b
            bag=temp

            p = np.bincount(train_y[bag])
            for i in range(6):
                if i >len(p)-1:
                    p=np.append(p,0)


            if len(pp)==0:
                    pp = np.array([p / np.sum(p)])
                    llp_x = np.array([new_train_x])
                # llp_x = np.array([train_x[bag]])
            else:
                # if np.array([p / np.sum(p)]).shape[0]!=1 or np.array([p / np.sum(p)]).shape[1]!=6:
                #     print(123)
                try:
                    pp = np.append(pp, np.array([p / np.sum(p)]), axis=0)
                    llp_x = np.append(llp_x, [new_train_x], axis=0)
                except:
                    a=1
                # llp_x = np.append(llp_x, [train_x[bag]], axis=0)

    distance=np.array(distance).sum()/len(distance)/o_config.bag_instance_num
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
    process()
