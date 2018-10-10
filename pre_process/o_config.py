import numpy as np
import math
class_num=2
prepared=0
typelist=[0,1,0,1,1,1,1,1,1,1,1,1,0,0]
# sub_k=280
data = 'instagram'
sub_k=0
pca_comp=50
n_iter=300#0#
epoch = 100
n_iter_without_progress = 100
bag_instance_num = 40  # 100
bag_num = 100  # 0#0
secret=np.array([0 for x in range(338)])
num_secret=10
threshold=10
lrate=0.01
bits = 1
f_threshold = 0.95
# agg_type='k_anony_partial'
agg_type = 'k_anony'
# agg_type='neighbour'
anony_k = 1
is_maliciou=1
T=10
a={1:2}
lamb = 1e-2  # 5
regularization = 'l1'
def find_sec(llp_x,secret):
    result=[]
    for bag in llp_x:
        for instance in bag:
            diff=instance-secret
            sum=[math.fabs(x) for x in diff]
            sum=np.sum(sum)
            if sum<threshold:
                result.append(instance)

    return result

def mal_encode(weight,secrets):

    temp = np.array([int(x * (10**bits)) for x in weight])
    sign=temp.copy()
    sign[sign>0]=1
    sign[sign<0]=-1
    temp = np.array([math.fabs(x) for x in temp])

    for i in range(1,len(secrets)+1):
        temp=temp*10+secrets[i-1]
    temp = temp * (0.1 ** (bits+len(secrets)))

    return temp*sign
