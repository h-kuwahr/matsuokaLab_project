# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D
import sys
import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
# from pybrain.datasets import SupervisedDataSet
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.supervised import BackpropTrainer

pow_threshold = float(10)
w_size = 300
s_size = 1
st_list = list([])
Test_NN = Sequential()
Test_NN.add(Dense(6, input_dim=11))
Test_NN.add(Activation('relu'))
Test_NN.add(Dense(1))
Test_NN.summary()
Test_NN.compile(optimizer='adam', loss='categorical_crossentropy')
Pfan_NN = Sequential()
Pfan_NN.add(Dense(6, input_dim=4))
Pfan_NN.add(Activation('relu'))
Pfan_NN.add(Dense(1))
Pfan_NN.compile(optimizer='adam', loss='categorical_crossentropy')


def getDiff(df, d, df_new):
    w_origin = d+datetime.timedelta(minutes=w_size)
    w = w_origin
    pow_diff = df.loc[d:w]
    while len(pow_diff) > 5:
        if max(pow_diff['Ps']) - min(pow_diff['Ps']) < pow_threshold:
            return (df_new, pow_diff.index[len(pow_diff.index) - 1])
        else:
            w = w - datetime.timedelta(minutes=s_size)
            pow_diff = df.loc[d:w]
    else:
        return(df_new.drop(pow_diff.index),
               pow_diff.index[len(pow_diff.index) - 1])


df = pd.read_csv('CFD_data_pivot.csv', index_col='truncated_dt')
header = df.columns.values.tolist()
Amb_temparature1 = df.loc[:, ['24144']]
Total_power1 = df.loc[:, ['24162']]
CPU_usage1 = df.loc[:, ['24163']]
Ffan1 = df.loc[:, ['24147']]
Ffan2 = df.loc[:, ['24155']]
Ffan3 = df.loc[:, ['24156']]
Ffan5 = df.loc[:, ['24157']]
Ffan6 = df.loc[:, ['24158']]
Ffan7 = df.loc[:, ['24159']]
Ffan8 = df.loc[:, ['24160']]
Ffan9 = df.loc[:, ['24161']]
Ffan10 = df.loc[:, ['24148']]
Ffan11 = df.loc[:, ['24149']]
Ffan12 = df.loc[:, ['24150']]
EXH_windspeed = df.loc[:, ['28596']]
Total_power1.describe()
# df_TaskCPU = CPU_usage1.loc[:, ['value']]/float(20)
# df_TaskCPU = df_TaskCPU.rename(columns={'value': 'CPU_usage'})
# df_TaskCPU.index = pd.to_datetime(CPU_usage1.loc[:, 'truncated_dt'])
# X, Y, Z = list([]), list([]), list([])
df_model = df.loc[:, ['24144', '24162', '24163', '28596']]
df_model_ip = df_model.interpolate()
# df_model_ip.fillna(method='bfill').plot(figsize=(32, 4), alpha=0.5)
df_model_std = df_model_ip.apply(lambda x: (x-x.mean())/x.std(), axis=0)
df_model_ip.plot(figsize=(32, 4), alpha=0.5)
df_model_std.index = pd.to_datetime(df_model_std.index)
dataset = df_model_std.loc['20170508115700':'20170519064200',
                           ['24144', '24162', '24163', '28596']]
print(dataset)
type(df_model_std.index[3])
# df_model.plot(figsize=(32, 4), alpha=5)
# CPU_usage1.hist(bins=1000)
# print(df_Fan)
# d = df_Ps.index[0]
# df_Ps_new = df_Ps
# while d < df_Ps.index[len(df_Ps.index) - 1]:
#     df_Ps_new, b = getDiff(df_Ps, d, df_Ps_new)
#     d = b + datetime.timedelta(minutes=1)
# df_Ps_new.plot(figsize=(32, 4), alpha=0.5)
# df_Ps.plot(figsize=(32, 4), alpha=0.5)
# print(df_Ps_new)
# df_Ffan2.plot(figsize=(32, 4), alpha=0.5)
# df_model.plot(figsize=(32, 4), alpha=0.5)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter3D(df_model['T_Ambient'], df_model['CPU_usage'], df_model['Ps'])
plt.show()
# print(header)
# print(data_server1)
# print(Amb_temparature1)
# print(Total_power1)
# print(CPU_usage1)
# print(df_Ps)
# print(df_At)
# print(df_TaskCPU)
# print(df_model)
