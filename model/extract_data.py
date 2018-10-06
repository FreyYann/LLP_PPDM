import pickle
import math
from src.pre_process import o_config
import numpy as np
from decimal import localcontext
from decimal import Decimal

result=[]
with open('/Users/yanxinzhou/course/thesis/is-FreyYann/docs/param/param.pkl', 'rb') as f:
    weights=pickle.load(f)

for weight in weights:
    # weight=[(x*(10**o_config.bits))% (10**o_config.bits) for x in weight]
    weight=[str(x*(10**o_config.bits)) for x in weight]
    weight=[x.split('.')[1] for x in weight]
    for i in range(o_config.num_secret):
        temp=[x[i:i+1] for x in weight]
        result.append([int(x) for x in temp])

np.savetxt('/Users/yanxinzhou/course/thesis/is-FreyYann/docs/param/result.out',np.array(result,dtype=np.int),delimiter=',',fmt='%.0f')