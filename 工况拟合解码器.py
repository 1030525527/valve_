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
h = Dense(30,activation='relu')(x_in)
h2 = Dense(100,activation='relu')(h)
h3 = Dense(50,activation='relu')(h2)
h4 = Dense(22)(h3)

epochs = 1000
lr = 0.001

# 建立模型

encoder = Model(x_in, h2, name='encoder')
encoder.compile(optimizer=Adam(lr=lr),loss= 'mse')
encoder.summary()


decoder = Model(h2, h4, name='decoder')
decoder.compile(optimizer=Adam(lr=lr),loss= 'mse')
decoder.summary()


sae =  Model(x_in,h4, name='sae')
sae.compile(optimizer=Adam(lr=lr),loss= 'mse')
sae.summary()
his1=sae.fit(x,u_new,
        shuffle=True,
        epochs=epochs)
loss_his = his1.history['loss']
u_pre = sae.predict(x)

u_pre2 = encoder.predict(x)
# u_value_new = np.zeros((55,22))


import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.subplot(211)
# plt.plot(np.arange(0, epochs), loss_his, label="mae")
# plt.subplot(212)
# plt.plot((u_pre[20,:].reshape(-1)))
# plt.plot((u_new[20,:]).reshape(-1))

plt.figure(2)
for i in range(11):
    plt.subplot(521)
    plt.plot((u_new[5*i, :]).reshape(-1))
    plt.subplot(522)
    plt.plot((u_pre[5 * i, :]).reshape(-1))
for i in range(11):
    plt.subplot(523)
    plt.plot((u_new[5*i+1, :]).reshape(-1))
    plt.subplot(524)
    plt.plot((u_pre[5 * i+1, :]).reshape(-1))
for i in range(11):
    plt.subplot(525)
    plt.plot((u_new[5*i+2, :]).reshape(-1))
    plt.subplot(526)
    plt.plot((u_pre[5 * i+2, :]).reshape(-1))
for i in range(11):
    plt.subplot(527)
    plt.plot((u_new[5*i+3, :]).reshape(-1))
    plt.subplot(528)
    plt.plot((u_pre[5 * i+3, :]).reshape(-1))
for i in range(11):
    plt.subplot(529)
    plt.plot((u_new[5*i+4, :]).reshape(-1))
    plt.subplot(5,2,10)
    plt.plot((u_pre[5 * i+4, :]).reshape(-1))


for i in range(55):
    for j in range(22):
        u_pre[i,j] = u_pre[i,j]*(u_m[j,0]-u_m[j,1]) + u_m[j,1]


u_pre = u_pre.reshape(11,-1)
s_pre = u_pre
io.savemat('E:\\健康基线数据2\\u_pre.mat', {'u_pre':u_pre})
'''---------------------------------------------------'''

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# from keras.layers import Input, Dense, Lambda
# from keras.models import Model
# from keras import backend as K
# import scipy.io as io
# from keras.optimizers import SGD,Adam
# x = np.zeros((55,3))
# x[0:5,0] = 900
# x[5:10,0] = 950
# x[10:15,0] = 1000
# x[15:20,0] = 1000
# x[20:25,0] = 1000
# x[25:30,0] = 1050
# x[30:35,0] = 1050
# x[35:40,0] = 1050
# x[40:45,0] = 1100
# x[45:50,0] = 1100
# x[50:55,0] = 1100

# x[0:5, 1] = 0
# x[5:10, 1] = 0
# x[10:15, 1] = 0
# x[15:20, 1] = 100
# x[20:25, 1] = 200
# x[25:30, 1] = 0
# x[30:35, 1] = 100
# x[35:40, 1] = 200
# x[40:45, 1] = 0
# x[45:50, 1] = 100
# x[50:55, 1] = 200

# for i in range(11):
#     x[i * 5, 2] = 1
#     x[i * 5 + 1, 2] = 2
#     x[i * 5 + 2, 2] = 3
#     x[i * 5 + 3, 2] = 4
#     x[i * 5 + 4, 2] = 5

# C = io.loadmat('E:\\健康基线数据2\\C_all.mat')['C_all']
# C_new = C.reshape(55,22,-1)

# x_m = np.zeros((3,2))
# for i in range(x.shape[1]):
#     xx_min = x[:,i].min()
#     xx_max = x[:,i].max()
#     x[:,i] = (x[:,i] - xx_min)/(xx_max - xx_min)
#     x_m[i, 0] = xx_max
#     x_m[i, 1] = xx_min

# c_m = np.zeros((2420,2))
# C_new = C_new.reshape(55,-1)
# for i in range(2420):
#     c_min = C_new[:,i].min()
#     c_max = C_new[:,i].max()
#     C_new[:, i] = (C_new[:, i] - c_min) / (c_max - c_min)
#     c_m[i, 0] = c_max
#     c_m[i, 1] = c_min

# C_new = C_new.reshape(55,22,-1)

# from keras.layers import Reshape, Conv2DTranspose,MaxPooling2D  #, Conv1DTranspose
# x_in = Input(shape=(3,))
# h = Dense(100, activation='relu')(x_in)
# h = Dense(1000, activation='relu')(h)
# h = Dense(2420, activation='relu')(h)

# h2= Reshape((22, 110, 1))(h)
# h3 = Conv2DTranspose(filters=64, kernel_size=6, activation='relu', strides=2, padding='same')(h2)
# hp1 = MaxPooling2D(pool_size=(2, 2),strides=None,padding='SAME', data_format=None)(h3)
# h4 = Conv2DTranspose(filters=64, kernel_size=6, activation='relu', strides=2, padding='same')(hp1)
# hp2 = MaxPooling2D(pool_size=(2, 2),strides=None,padding='SAME', data_format=None)(h4)
# h5 = Conv2DTranspose(filters=1, kernel_size=6, activation='relu', padding='same')(hp2)
# # hp3 = MaxPooling2D(pool_size=(2, 2),strides=None,padding='SAME', data_format=None)(h5)

# h6 = Reshape((22, 110))(h5)

# epochs = 1000
# lr = 0.001

# # 建立模型
# decoder = Model(x_in, h6, name='decoder')
# decoder.compile(optimizer=Adam(lr=lr),loss= 'mse')
# decoder.summary()
# his1 = decoder.fit(x,C_new,
#         shuffle=True,
#         epochs=epochs)
# loss_his = his1.history['loss']

# C_pre = decoder.predict(x)
# C_new = C.reshape(55,22,-1)

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns

# plt.figure()
# plt.plot(np.arange(0, epochs), loss_his, label="mae")

# for j in range(5):
#     plt.figure(j+2)
#     for i in range(11):
#         plt.subplot(11,2,i*2+1)
#         sns.heatmap(C_new[i*5+j,:,:])
#         plt.subplot(11,2,i*2+2)
#         sns.heatmap(C_pre[i*5+j,:,:])

# C_pre = C_pre.reshape(55,-1)
# for i in range(2420):
#     C_pre[:,i] = C_pre[:,i]*(c_m[i,0] - c_m[i,1]) + c_m[i,1]
# C_pre = C_pre.reshape(55,22,-1)
# C_pre = C_pre.reshape(11,110,110)

# io.savemat('E:\\健康基线数据2\\C_pre.mat',{'C_pre': C_pre})  # write
