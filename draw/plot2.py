import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}
plt.figure(figsize=(10, 8))
matplotlib.rc('font', **font)
import numpy as np

sns.set_style("darkgrid", {"xtick.major.size": 1, "ytick.major.size": 1})
plt.xlabel("random noise: k'= 1000", fontdict=None, labelpad=None)
# plt.xlabel('random noise: epsilon= 0.1', fontdict=None, labelpad=None)
#
# sensitivity=np.array([1 /(250 *x) for x in range(1, 5)])
sensitivity = 1 / 1000
epsilon = np.array([1 / (10 * x) for x in range(1, 5)])
# epsilon=0.1

# for s in sensitivity:
#     scale = s / epsilon
#     noise=np.random.laplace(0,scale,1000)
#     sns.distplot(noise,hist=False,label="k': {}".format(round(1/s,3)))
for e in epsilon:
    scale = sensitivity / e
    noise = np.random.laplace(0, scale, 1000)
    sns.distplot(noise, hist=False, label="epsilon: {}".format(round(e, 3)))
plt.legend()
plt.savefig('/Users/yanxinzhou/course/thesis/is-FreyYann/data/pic/5.png')
plt.show()
