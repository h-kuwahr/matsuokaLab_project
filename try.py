# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
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


df = pd.read_csv('CFD_data_org.csv')
header = df.columns.values.tolist()
data_server1 = df[df['hostid'] == 10128]
Amb_temparature1 = df[df['itemid'] == 24144]
Total_power1 = df[df['itemid'] == 24162]
CPU_usage1 = df[df['itemid'] == 24163]
Ffan1 = df[df['itemid'] == 24147]
Ffan2 = df[df['itemid'] == 24155]
Ffan3 = df[df['itemid'] == 24156]
Ffan5 = df[df['itemid'] == 24157]
Ffan6 = df[df['itemid'] == 24158]
Ffan7 = df[df['itemid'] == 24159]
Ffan8 = df[df['itemid'] == 24160]
Ffan9 = df[df['itemid'] == 24161]
Ffan10 = df[df['itemid'] == 24148]
Ffan11 = df[df['itemid'] == 24149]
Ffan12 = df[df['itemid'] == 24150]
EXH_windspeed = df[df['itemid'] == 28596]
df_TaskCPU = CPU_usage1.loc[:, ['value']]/float(20)
df_TaskCPU = df_TaskCPU.rename(columns={'value': 'CPU_usage'})
df_TaskCPU.index = pd.to_datetime(CPU_usage1.loc[:, 'truncated_dt'])
X, Y, Z = list([]), list([]), list([])
df_Ps = Total_power1.loc[:, ['value']]
df_Ps = df_Ps.rename(columns={'value': 'Ps'})
df_Ps.index = pd.to_datetime(Total_power1.loc[:, 'truncated_dt'])
df_At = Amb_temparature1.loc[:, ['value']]
df_At = df_At.rename(columns={'value': 'T_Ambient'})
df_At.index = pd.to_datetime(Amb_temparature1.loc[:, 'truncated_dt'])
df_model = df_Ps.join(df_At.join(df_TaskCPU, how='inner'), how='inner')
df_Ffan1 = Ffan1.loc[:, ['value']]
df_Ffan1 = df_Ffan1.rename(columns={'value': 'Ffan1'})
df_Ffan1.index = pd.to_datetime(Ffan1.loc[:, 'truncated_dt'])
df_Ffan2 = Ffan2.loc[:, ['value']]
df_Ffan2 = df_Ffan2.rename(columns={'value': 'Ffan2'})
df_Ffan2.index = pd.to_datetime(Ffan2.loc[:, 'truncated_dt'])
df_Ffan3 = Ffan3.loc[:, ['value']]
df_Ffan3 = df_Ffan3.rename(columns={'value': 'Ffan3'})
df_Ffan3.index = pd.to_datetime(Ffan3.loc[:, 'truncated_dt'])
df_Ffan5 = Ffan5.loc[:, ['value']]
df_Ffan5 = df_Ffan5.rename(columns={'value': 'Ffan5'})
df_Ffan5.index = pd.to_datetime(Ffan5.loc[:, 'truncated_dt'])
df_Ffan6 = Ffan6.loc[:, ['value']]
df_Ffan6 = df_Ffan6.rename(columns={'value': 'Ffan6'})
df_Ffan6.index = pd.to_datetime(Ffan6.loc[:, 'truncated_dt'])
df_Ffan7 = Ffan7.loc[:, ['value']]
df_Ffan7 = df_Ffan7.rename(columns={'value': 'Ffan7'})
df_Ffan7.index = pd.to_datetime(Ffan7.loc[:, 'truncated_dt'])
df_Ffan8 = Ffan8.loc[:, ['value']]
df_Ffan8 = df_Ffan8.rename(columns={'value': 'Ffan8'})
df_Ffan8.index = pd.to_datetime(Ffan8.loc[:, 'truncated_dt'])
df_Ffan9 = Ffan9.loc[:, ['value']]
df_Ffan9 = df_Ffan9.rename(columns={'value': 'Ffan9'})
df_Ffan9.index = pd.to_datetime(Ffan9.loc[:, 'truncated_dt'])
df_Ffan10 = Ffan10.loc[:, ['value']]
df_Ffan10 = df_Ffan10.rename(columns={'value': 'Ffan10'})
df_Ffan10.index = pd.to_datetime(Ffan10.loc[:, 'truncated_dt'])
df_Ffan11 = Ffan11.loc[:, ['value']]
df_Ffan11 = df_Ffan11.rename(columns={'value': 'Ffan11'})
df_Ffan11.index = pd.to_datetime(Ffan11.loc[:, 'truncated_dt'])
df_Ffan12 = Ffan12.loc[:, ['value']]
df_Ffan12 = df_Ffan12.rename(columns={'value': 'Ffan12'})
df_Ffan12.index = pd.to_datetime(Ffan12.loc[:, 'truncated_dt'])
df_EW = EXH_windspeed.loc[:, ['value']]
df_EW.index = pd.to_datetime(EXH_windspeed.loc[:, 'truncated_dt'])
df_Fan = df_Ffan1.join(df_Ffan2, how='inner')
df_Fan = df_Fan.join(df_Ffan3, how='inner')
df_Fan = df_Fan.join(df_Ffan5, how='inner')
df_Fan = df_Fan.join(df_Ffan6, how='inner')
df_Fan = df_Fan.join(df_Ffan7, how='inner')
df_Fan = df_Fan.join(df_Ffan8, how='inner')
df_Fan = df_Fan.join(df_Ffan9, how='inner')
df_Fan = df_Fan.join(df_Ffan10, how='inner')
df_Fan = df_Fan.join(df_Ffan11, how='inner')
df_Fan = df_Fan.join(df_Ffan12, how='inner')
df_model = df_model.join(df_Fan, how='inner')
# print(df_model)
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
df_EW.plot(figsize=(64, 4), alpha=0.5)
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
