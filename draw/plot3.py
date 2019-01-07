import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)
k = [50, 100, 250, 500, 750, 1000]
LR = []
e01 = [0.82, 0.81, 0.76, 0.79, 0.83, 0.79]
e05 = [.78, 0.81, 0.78, 0.78, 0.82, 0.82]
e001 = [.74, .80, .81, .80, .82, .81]
in1 = [0.087, 0.063, 0.051, .047, .045, .043]
in2 = [0.15, .092, .064, .053, .048, .047]
in3 = [.63, .34, .17, .11, .08, .074]
plt.figure(figsize=(10, 8))
sns.set_style("darkgrid", {"xtick.major.size": 1, "ytick.major.size": 1})

# plt.xlabel('K-anonymous', fontdict=None, labelpad=None)
plt.ylabel('K-anonymous', fontdict=None, labelpad=None)
plt.xlabel('Infomation loss', fontdict=None, labelpad=None)
# plt.ylabel('F1-score', fontdict=None, labelpad=None)

plt.plot(k, in1, label="Epsilon 0.1")
plt.plot(k, in2, label="Epsilon 0.05")
plt.plot(k, in3, label="Epsilon 0.01")
#
# plt.plot(  k,e01,label="Epsilon 0.1")
# plt.plot( k, e05,label="Epsilon 0.05")
# plt.plot(k,e001, label="Epsilon 0.01")

plt.legend()
# plt.show()
plt.savefig('/Users/yanxinzhou/course/thesis/is-FreyYann/data/pic/8.png')
