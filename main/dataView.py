import pickle
import numpy as np

with open('/Users/yanxinzhou/course/thesis/is-FreyYann/data/adult/train.pkl', 'rb') as f:
    df = pickle.load(f)
neg=np.where(df[:,-1]==0)
pos=np.where(df[:,-1]==1)

df_n=df[neg].sum(axis=0)/df[neg].shape[0]
df_p=df[pos].sum(axis=0)/df[pos].shape[0]

diff=df_n-df_p
diff=np.array([round(x,2) for x in diff])