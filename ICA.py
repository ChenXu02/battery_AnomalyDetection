from scipy.integrate import simpson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def moving_average(interval, windowsize,mark):#滑动平均
    if mark==1:
        re = np.ndarray(interval.shape)
        interval = np.concatenate((interval, np.tile(interval[[-1],:], (windowsize - 1,1))),axis=0)
        for i in range(interval.shape[1]):
            window = np.ones(int(windowsize)) / float(windowsize)
            convRe= np.convolve(interval[:,i], window, 'valid')
            re[:, i]=convRe
    else:
        interval = np.hstack((interval, np.tile(interval[[-1]], windowsize - 1)))
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, 'valid')
    return re
def Diff(data):#差分
    data_diff = np.ndarray(data.shape)
    for i in range(data.shape[0] - 1):
        data_diff[i]=data[i + 1] - data[i]
    data_diff[-1]=data_diff[-2]
    return data_diff
def ICA(V,I):#计算Q值
    # 求积分部分
    V=np.array(V)
    I=np.array(I)
    Q=np.ndarray(I.shape)
    for i in range(0,V.shape[0]):
        time = np.arange(i+1)
        Q[i]=simpson(I[0:i+1], time)
    D_Q=np.tile(moving_average(Diff(Q),10,0).reshape(-1,1)/20000,16)
    D_V=moving_average(Diff(V),30,0)
    #print(D_Q,'dq')
    #print(D_Q[0][0],D_V[0][0],D_Q[0][0]/D_V[0][0],(D_Q/D_V)[0][0],'dv')
    return 0,Q/1000000
    # > 344.79333333333
