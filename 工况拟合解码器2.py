# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 22:27:28 2023

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import scipy.io as io
from keras.optimizers import SGD,Adam

x = np.zeros((55,3))
x[0:5,0] = 900
x[5:10,0] = 950
x[10:15,0] = 1000
x[15:20,0] = 1000
x[20:25,0] = 1000
x[25:30,0] = 1050
x[30:35,0] = 1050
x[35:40,0] = 1050
x[40:45,0] = 1100
x[45:50,0] = 1100
x[50:55,0] = 1100

x[0:5, 1] = 0
x[5:10, 1] = 0
x[10:15, 1] = 0
x[15:20, 1] = 100
x[20:25, 1] = 200
x[25:30, 1] = 0
x[30:35, 1] = 100
x[35:40, 1] = 200
x[40:45, 1] = 0
x[45:50, 1] = 100
x[50:55, 1] = 200

for i in range(11):
    x[i * 5, 2] = 1
    x[i * 5 + 1, 2] = 2
    x[i * 5 + 2, 2] = 3
    x[i * 5 + 3, 2] = 4
    x[i * 5 + 4, 2] = 5

u = io.loadmat('E:\\健康基线数据2\\u_all.mat')['u_all']
# u = io.loadmat('E:\\健康基线数据2\\s_all.mat')['s_all']
u_new = np.zeros((55,22))
for i in range(11):
    for j in range(5):
        u_new[i*5+j,:] = u[i,22*j:22*(j+1)]

x_m = np.zeros((3,2))
for i in range(x.shape[1]):
    xx_min = x[:,i].min()
    xx_max = x[:,i].max()
    x[:,i] = (x[:,i] - xx_min)/(xx_max - xx_min)
    x_m[i, 0] = xx_max
    x_m[i, 1] = xx_min

u_m = np.zeros((22,2))
for i in range(u_new.shape[1]):
    xx_min = u_new[:,i].min()
    xx_max = u_new[:,i].max()
    u_new[:,i] = (u_new[:,i] - xx_min)/(xx_max - xx_min)
    u_m[i, 0] = xx_max
    u_m[i, 1] = xx_min

x_in = Input(shape=(3,))
h = Dense(30, activation='relu')(x_in)
h2 = Dense(100, activation='relu')(h)
h3 = Dense(50, activation='relu')(h2)

# Encoder
encoded = Dense(10, activation='relu')(h3)
encoder = Model(x_in, encoded, name='encoder')
encoder.summary()

# Decoder
decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(22)(decoded)
decoder = Model(encoded, decoded, name='decoder')
decoder.summary()

# Stacked Autoencoder
sae = Model(x_in, decoder(encoder(x_in)), name='sae')
sae.summary()

# Compile and train the model
epochs = 1000
lr = 0.001
sae.compile(optimizer=Adam(lr=lr), loss='mse')
his1 = sae.fit(x, u_new, shuffle=True, epochs=epochs)

u_pre = sae.predict(x)

u_pre2 = encoder.predict(x)
# u_value_new = np.zeros((55,22))
import matplotlib.pyplot as plt

loss_values = his1.history['loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(u_new, u_pre)
print(f"均方误差（MSE）：{mse}")
# 使用编码器模型对输入 x 进行编码
u_pre2 = encoder.predict(x)

# 进行聚类分析
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)  # 设置聚类簇数
clusters = kmeans.fit_predict(u_pre2)

# 可视化聚类结果
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 选择前三个编码维度进行可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(u_pre2[:, 0], u_pre2[:, 1], u_pre2[:, 2], c=clusters)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# # plt.subplot(211)
# # plt.plot(np.arange(0, epochs), loss_his, label="mae")
# # plt.subplot(212)
# # plt.plot((u_pre[20,:].reshape(-1)))
# # plt.plot((u_new[20,:]).reshape(-1))

# plt.figure(2)
# for i in range(11):
#     plt.subplot(521)
#     plt.plot((u_new[5*i, :]).reshape(-1))
#     plt.subplot(522)
#     plt.plot((u_pre[5 * i, :]).reshape(-1))
# for i in range(11):
#     plt.subplot(523)
#     plt.plot((u_new[5*i+1, :]).reshape(-1))
#     plt.subplot(524)
#     plt.plot((u_pre[5 * i+1, :]).reshape(-1))
# for i in range(11):
#     plt.subplot(525)
#     plt.plot((u_new[5*i+2, :]).reshape(-1))
#     plt.subplot(526)
#     plt.plot((u_pre[5 * i+2, :]).reshape(-1))
# for i in range(11):
#     plt.subplot(527)
#     plt.plot((u_new[5*i+3, :]).reshape(-1))
#     plt.subplot(528)
#     plt.plot((u_pre[5 * i+3, :]).reshape(-1))
# for i in range(11):
#     plt.subplot(529)
#     plt.plot((u_new[5*i+4, :]).reshape(-1))
#     plt.subplot(5,2,10)
#     plt.plot((u_pre[5 * i+4, :]).reshape(-1))


# for i in range(55):
#     for j in range(22):
#         u_pre[i,j] = u_pre[i,j]*(u_m[j,0]-u_m[j,1]) + u_m[j,1]


# u_pre = u_pre.reshape(11,-1)
# s_pre = u_pre
# io.savemat('E:\\健康基线数据2\\u_pre.mat', {'u_pre':u_pre})