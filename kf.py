import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn import preprocessing
from sklearn.metrics import r2_score
import math



# # 4_1 Q=1
# n = 0.63316574
# A = 0.56429719
# filename = 'data/data4_1.csv'
# data_count = 30

# # 4_2 Q=1
# n = 0.61663607
# A = 0.52466605
# filename = 'data/data4_2.csv'
# data_count = 30


# 4_3 Q=1
n = 0.617579
A = 0.58453867
filename = 'data/data4_3.csv'
data_count = 30


# # 5_1
# n = 0.92346027
# A = 0.64156834
# filename = 'data5_1.csv'
# data_count = 28

# # 5_2
# n = 0.68981584
# A = 1.94297796
# filename = 'data5_2.csv'
# data_count = 28

# # 5_3
# n = 0.4736327
# A = 7.48884673
# filename = 'data5_3.csv'
# data_count = 28

# # 5_4
# n = 0.47944589
# A = 4.5557607
# filename = 'data5_4.csv'
# data_count = 28

# # 5_5
# n = 0.91620582
# A = 1.03756635
# filename = 'data5_5.csv'
# data_count = 28

# # 5_6
# n = 1.04121841
# A = 0.7511848
# filename = 'data5_6.csv'
# data_count = 28

# # 5_7
# n = 0.73252593
# A = 2.15044521
# filename = 'data5_7.csv'
# data_count = 28

# # 5_8
# n = 0.41235549
# A = 5.55203983
# filename = 'data5_8.csv'
# data_count = 28

# new data55
# n = 0.90396871
# A = 0.08034555
# filename = '55(1).csv'
# data_count = 55

# # new data60
# n = 0.93118298
# A = 0.07061449
# filename = '60.csv'
# data_count = 60


data = pd.read_csv(filename)
X = data['Time/h']
Y = data['weight']
X = X.values
H = data['Time/h']
H = np.mat(H)

u = np.mat(math.pow(12, n))
X = np.mat(X)
Y = np.mat(Y)

# print(X.shape)
# print(X)
# print(Y)

z_watch = Y
# print(z_watch)


# 创建一个方差为1的高斯噪声，精确到小数点后两位
noise = np.round(np.random.normal(0, 0.0001, data_count), 2)
noise_mat = np.mat(noise)

# 将z的观测值和噪声相加
# print(z_watch)
# print(noise_mat)
z_mat = z_watch + noise_mat
# print(z_mat)

# 定义x的初始状态
x_mat = np.mat([0, ])
# 定义初始状态协方差矩阵
p_mat = np.mat([1])
# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.mat([1])
# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([0.5])
# 定义观测矩阵
h_mat = np.mat([1])
# 定义观测噪声协方差
r_mat = np.mat([1])
# 定义控制矩阵
b_mat = np.mat([A])
record = []
record_x =[]
record_predict = []
record_fix = []
record_z = []
for i in range(data_count):
    x_predict = f_mat * x_mat + b_mat * u.T
    record_predict.append(x_predict[0, 0])
    p_predict = f_mat * p_mat * f_mat.T + q_mat
    kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
    x_mat = x_predict + kalman * (z_mat[0, i] - h_mat * x_predict)
    record_fix.append(x_mat[0, 0])
    record_z.append(z_mat[0, i])
    p_mat = (np.eye(1) - kalman * h_mat) * p_predict
    # print(x_predict[0, 0], x_mat[0, 0])
    print(x_mat[0, 0])
    record.append(x_mat[0, 0])
    # plt.plot(H[0, i], x_mat[0, 0], 'ro', markersize=1)
    # plt.plot(H[0, i], z_mat[0, i], 'bo', markersize=1)
    # plt.plot(H[0, i], x_predict[0, 0], 'go', markersize=1)
# a=np.array(H[0])
# a=a.reshape(1,-1)
# plt.figure(a[0], record_fix)
# plt.plot(a[0], record_fix, 'ro', markersize=1)
# plt.plot(a[0], record_z, 'bo', markersize=1)
# plt.plot(a[0], record_predict, 'go', markersize=1)
# plt.show()
plt.xlabel("Time(h)")
plt.ylabel("Weight gain(g·m-2)")
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(range(0, len(record_predict)*12, 12), record_predict, label = 'Measurements')
plt.plot(range(0, len(record_predict)*12, 12), np.array(record_fix), label = 'Kalman Filter Prediction')
plt.plot(range(0, len(record_predict)*12, 12), record_z, label = 'Real statement' )
plt.legend()
plt.show()

print(type(record_predict))
print(type(record_fix))
print(type(record_z))


# a=np.array(H[0])
# a=a.reshape(1,-1)
# print(a[0])
# print(record_fix)

y_real = z_mat
y_pre1 = np.mat(record)
print(r2_score(np.array(y_real)[0], np.array(y_pre1)[0]))
# plt.show()





# data = pd.read_csv("data41.csv")
# X = data['Time/h']
# Y = data['Added Weight']
# H = data['Time/h']
# # H = H.values
# # H = np.log(H)
# one = np.ones([1, 28])
#
# X = X.values
# # X = np.vstack((X, one))
# u = np.vstack((np.log(X), one))
# u = np.mat(u)
# X = np.mat(X)
# Y = np.mat(Y)
# H = np.mat(H)
# # print(X.shape)
# # print(X)
# # print(Y)
#
# z_watch = Y
# # print(z_watch)
#
#
# # 创建一个方差为1的高斯噪声，精确到小数点后两位
# noise = np.round(np.random.normal(0, 0.0001, 28), 2)
# noise_mat = np.mat(noise)
#
# # 将z的观测值和噪声相加
# z_mat = z_watch + noise_mat
# # print(z_mat)
#
# # 定义x的初始状态
# x_mat = np.mat([0, ])
# # 定义初始状态协方差矩阵
# p_mat = np.mat([1])
# # 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
# f_mat = np.mat([0])
# # 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
# q_mat = np.mat([0.01])
# # 定义观测矩阵
# h_mat = np.mat([1])
# # 定义观测噪声协方差
# r_mat = np.mat([1])
# # 定义控制矩阵
# b_mat = np.mat([-0.20358046, 2.52076781])
# record = []
# for i in range(28):
#     x_predict = f_mat * x_mat + b_mat * u.T[i].T
#     p_predict = f_mat * p_mat * f_mat.T + q_mat
#     kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
#     x_mat = x_predict + kalman * (z_mat[0, i] - h_mat * x_predict)
#     p_mat = (np.eye(1) - kalman * h_mat) * p_predict
#     # print(np.exp(x_predict[0, 0]), np.exp(x_mat[0, 0]))
#     print(np.exp(x_mat[0, 0]))
#     record.append(np.exp(x_mat[0, 0]))
#     plt.plot(H[0, i], np.exp(x_mat[0, 0]), 'ro', markersize=1)
#     plt.plot(H[0, i], z_mat[0, i], 'bo', markersize=1)
#     plt.plot(H[0, i], np.exp(x_predict[0, 0]), 'go', markersize=1)
#
# y_real = z_mat
# y_pre = np.mat(record)
# # SSR = 0
# # SST = 0
# # for i in range(30):
# #     SSR = SSR + (y_pre[0, i] - np.mean(y_pre)) * (y_pre[0, i] - np.mean(y_pre))
# #     SST = SST + (y_real[0, i] - np.mean(y_real)) * (y_real[0, i] - np.mean(y_real))
# # print(y_real)
# # print(y_pre)
# # print(SSR/SST)
# # print(np.array(y_real)[0].shape)
# print(r2_score(np.array(y_real)[0], np.array(y_pre)[0]))
# plt.show()
#
