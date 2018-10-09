import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.xlabel('k-anonymous', fontdict=None, labelpad=None)
# plt.xlabel('infomation loss', fontdict=None, labelpad=None)
plt.ylabel('F1-score', fontdict=None, labelpad=None)

info_loss0 = [0, 8.03, 9.02, 9.59, 9.70]
k = [1, 3, 5, 10, 15]
F1_0 = [0.86, 0.84, .82, 0.78, 0.73]
plt.plot(k, F1_0, label="logistic_regression")

info_loss1 = [0, 5.28, 6.43, 7.36, 7.70]
F11 = [0.81, 0.81, .81, 0.81, 0.81]
plt.plot(k, F11, label="Neighbour_K_anonymous")

info_loss2 = [0, 8.02, 9.23, 9.82, 9.943]
F12 = [0.81, 0.81, .79, 0.74, 0.74]
plt.plot(k, F12, label="Random_K_anonymous")

info_loss3 = [0, 2.24, 4.00, 4.85, 5.97, 9.28]
k3 = [1, 3, 5, 8, 10, 15]
F1_3 = [0.75, 0.75, .75, .74, 0.74, 0.74]
plt.plot(k3, F1_3, label="Patial_anonymous")

plt.figure(figsize=(12, 8))
plt.xlabel('k-anonymous', fontdict=None, labelpad=None)
plt.ylabel('infomation loss', fontdict=None, labelpad=None)
# plt.ylabel('F1-score', fontdict=None, labelpad=None)
plt.plot(k, info_loss0, label="logistic_regression")
plt.plot(k, info_loss1, label="Neighbour_K_anonymous")
plt.plot(k, info_loss2, label="Random_K_anonymous")
plt.plot(k3, info_loss3, label="Patial_anonymous")

plt.legend()
plt.show()