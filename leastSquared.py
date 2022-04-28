import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import datasets, linear_model
import math

# log∆W=logA+nlogN

# 导入数据
data = pd.read_csv("data/data4_3.csv")
X = data['Time/h']
X = X.values
X_log = np.log(X)
Y = data['weight']
Y = Y.values
Y_log = np.log(Y)
# print(X)
# print(Y)
# print(type(X))
# print(type(X))

# dataframe = pd.DataFrame({'Time': X, 'Added weight': Y})
# # 将DataFrame存储为csv,mode='a'表示追加写入
# dataframe.to_csv("data_57.csv", mode='w', header=True, sep=',', index=0)

# 线性回归模型
regr = linear_model.LinearRegression()
regr.fit(X_log.reshape(-1, 1), Y_log.reshape(-1, 1))
plt.xlabel("log_Time(log(h))")
plt.ylabel("log_Weight(log(g·m-2))")

print("b:", regr.intercept_)
print("k:", regr.coef_)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


plt.scatter(X_log, Y_log, s=10, color='#00CED1', marker='s')
plt.plot(X_log, regr.predict(X_log.reshape(-1, 1)), color='#DC143C', linewidth=2)
# plt.scatter(X, Y, s=10, color='black', marker='s')
# plt.plot(X, np.exp(regr.predict(X_log.reshape(-1, 1))), color='black', linewidth=2)
xlocs = []
tx = 1.5
for i in range(9):
    tx = tx + 1.0 / 2
    xlocs.append(tx)

ylocs = []
ty = 0
for i in range(7):
    ty = ty + 1.0 / 2
    ylocs.append(ty)
# print(xlocs)
# print(ylocs)

plt.xticks(xlocs)
plt.yticks(ylocs)

plt.show()

b = regr.intercept_
k = regr.coef_
# n = k - 1
# A = np.exp(b)
n = k
A = np.exp(b)
print("n:", n)
print("A:", A)

# time_supplement = np.array([18, 42, 66, 90, 114, 138, 162])
# added_weight_supplement = regr.predict(np.log(time_supplement).reshape(-1, 1))
# added_weight_supplement = np.exp(added_weight_supplement)
# # 字典中的key值即为csv中列名
# for i in range(7):
#     dataframe_supplement = pd.DataFrame({'Time/h': time_supplement[i], 'Added weight': added_weight_supplement[i]})
#     # 将DataFrame存储为csv,mode='a'表示追加写入
#     dataframe_supplement.to_csv("data5_8.csv", mode='a', header=False, sep=',', index=0)
