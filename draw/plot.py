import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)

plt.figure(figsize=(10, 8))
sns.set_style("darkgrid", {"xtick.major.size": 1, "ytick.major.size": 1})
plt.xlabel('k-anonymous', fontdict=None, labelpad=None)
# plt.xlabel('infomation loss', fontdict=None, labelpad=None)
plt.ylabel('F1-score', fontdict=None, labelpad=None)

k = [1, 3, 5, 10, 20]

info_loss1 = [0, 0.031, 0.038, 0.044, 0.046]
F11 = [0.81, 0.79, .80, 0.81, 0.80]
plt.plot(k, F11, label="K Nearest Aggregation")

info_loss2 = [0, 0.032, 0.039, 0.044, 0.046]
F12 = [0.81, 0.80, .81, 0.80, 0.81]
plt.plot(k, F12, label="Random K Aggregation")

info_loss3 = [0, 0.018, 0.022, 0.025, 0.028, 0.030, 0.032]
k3 = [1, 3, 5, 10, 20, 30, 50]
F1_3 = [0.81, 0.82, .82, .81, .83, .83, .83]
plt.plot(k3, F1_3, label="K Tree Aggregation")

info_loss0 = [0, 0.013, 0.016, 0.019, 0.022, 0.023, 0.025]
F1_0 = [0.86, 0.85, .84, 0.79, 0.74, 0.74, 0.74]
plt.plot(k3, F1_0, label="Logistic_regression")

plt.legend()
plt.savefig('/Users/yanxinzhou/course/thesis/is-FreyYann/data/pic/1.png')
plt.show()

plt.figure(figsize=(10, 8))
sns.set_style("darkgrid", {"xtick.major.size": 1, "ytick.major.size": 1})

plt.ylabel('K-anonymous', fontdict=None, labelpad=None)
plt.xlabel('Infomation loss', fontdict=None, labelpad=None)
# plt.ylabel('F1-score', fontdict=None, labelpad=None)
plt.plot(info_loss1, k, label="K Nearest Aggregation")
plt.plot(info_loss2, k, label="Random K Aggregation")
plt.plot(info_loss3, k3, label="K Tree Aggregation")
plt.plot(info_loss0, k3, label="Logistic_regression")

plt.legend()
# plt.show()
plt.savefig('/Users/yanxinzhou/course/thesis/is-FreyYann/data/pic/2.png')
