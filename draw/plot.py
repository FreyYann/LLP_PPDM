import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.xlabel('infomation loss', fontdict=None, labelpad=None)
plt.ylabel('F1-score', fontdict=None, labelpad=None)

info_loss0=[0,1.2,3.00,4.5,5.26,8,9.6]
F1_0=[0.86,0.8,.70,0.75,0.77,0.74,0.74]
plt.plot(info_loss0, F1_0, label="logistic_regression")

info_loss1=[0,2.24,4.00,4.85,5.97,9.28]
F1_1=[0.75,0.75,.75,.74,0.74,0.74]
plt.plot(info_loss1, F1_1, label="Patial_anonymous")

info_loss2=[0,7.65,9,9.3]
F1_2=[0.75,0.75,.74,0.74]
plt.plot(info_loss2, F1_2, label="K_anonymous")
# plt.plot(x_plot, prec_plot, label="Precision")
# plt.plot(x_plot, rec_plot, label="Recall")

plt.legend()
plt.show()