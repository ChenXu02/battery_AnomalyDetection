
import csv
from numpy import *
import torch
import glob
import numpy as np
import re
import os
import xlrd
import time
import pandas as pd
atime=time.time()

def SG(file):#C,T
    datas = np.load(file)
    u = datas['u']
    s = datas['s']
    th = datas['th']
    return u,s,th


def get_mahalanobis(D, u, x):

    invD = np.linalg.inv(D)  # 协方差逆矩阵
    tp = np.array(x)-np.array(u)
    return dot(dot(tp, invD), tp.T)



df = pd.read_excel('sample.xlsx', '1', header=1)
data_array = np.array(df)
#print(data_array)

path = '/Users/didi/Desktop/电池数据/test/data/'
files = glob.glob(path + "/*")
# test = Judge("./formula")
# test.show()

k=0
chargeTotal=0
ab=0
total=0
realab=0
realn=0
labab=0
labn=0

fakedata=0

originalData=[]
datam_p=[]
chargeType='BAAC9'

ID_p = 0  # 上一个电池的ID
STD_record = []  # 记录电压
I_record = []  # 记录电流
T_record=[]
SOC0 = -1
SOC50 = -1
SOC70 = -1
ab_c=0
ar=4
file1='paraSave_BAAC9_0.0{}.npz' .format(ar)
file2='paraSave_BAAC9_0.0{}.npz'.format(ar+1)
u, s, th1 = SG(file1)
_, _, th2 = SG(file2)
print(u,s,th1)




for file_name in os.listdir(path):
    print(file_name)
    if ".csv" not in file_name:
        continue
    open_file = path + file_name
    with open(open_file, 'r',encoding='gb2312') as f1:
        reader = csv.reader(f1)
        for row in reader:
            if len(row) < 42:
                continue
            ID = row[39]
            time=row[0]
            if ID[0:5] != chargeType:
                continue
            if not row[42].isdigit():  # 去除null
                #print(ID)
                continue
            I = float(row[42])
            T = float(row[25])  # 环境温度
            C=float(row[38])
            SOC = float(row[4])
            originalData.append(row)

            if ID != ID_p:  # 到达新的ID
                originalData = []
                ID_p = ID
                SOC0 = -1
                SOC50 = -1
                SOC70 = -1
                T0 = T
                '''
                if ID=='BCB9921031500169':
                    print(T0)
                    print(row)
                    print(list(map(float, row[5:21])))
                    print(file_name)
                    print(0 not in list(map(float, row[5:21])))
                    time.sleep(100)
                '''
                if I >= 0 and SOC0 == -1:
                    v1=list(map(float, row[5:21]))
                    SOC0 = np.std(list(map(float, row[5:21])))

            if SOC == 50 and SOC50 == -1:
                SOC50 = np.std(list(map(float, row[5:21])))
            if SOC == 70 and SOC70 == -1:
                SOC70=1
                if SOC50 != -1:
                    SOC50 = (SOC50 + np.std(list(map(float, row[5:21])))) / 2

                else:
                    SOC50 = np.std(list(map(float, row[5:21])))

                P = [SOC0, SOC50]
                s[0][0] = s[0][0] /( 10.01 / (T0 / 2000 + 10))*0.992
                s[1][1]=s[1][1]*10.01/(T0/2000+10)*0.992
                dis = get_mahalanobis(s, u, P)

                total += 1
                if (   dis >= th2 and T0.is_integer() and (0 not in v1)):
                    #print(dis)
                    adata = [file_name]+[ID] +[C]+ [row[43]] + [SOC0] + [SOC50] +[T0]+ [dis] + ['abnormal']
                    ab += 1

                else:
                    adata = [file_name] + [ID] + [C] + [row[43]] + [SOC0] + [SOC50]+[T0] + [dis] + ['normal']

                for d in range(0,len(data_array)):
                    #print(str(time),str(data_array[d][1])[0:10])
                    if ID==data_array[d][0] and str(time)==str(data_array[d][1])[0:10]:
                        adata+=[data_array[d][3]]

                        if adata[-1] == 'abnormal':
                            labab += 1
                        else:
                            labn += 1
                        if adata[-1]==adata[-2]:
                            ab_c += 1
                            if adata[-1]=='abnormal':
                                realab+=1
                            else:
                                realn+=1
                        else:
                            for ii in range(0, len(originalData)):
                                with open('test2BAAC9_original_{}.csv'.format(ar), 'a+', newline='') as fori:
                                    writer1 = csv.writer(fori)
                                    writer1.writerow(originalData[ii]+[adata[-2]])
                        break

                with open('test2BAAC9_T_{}.csv'.format(ar), 'a+', newline='') as fth:
                    writer = csv.writer(fth)
                    writer.writerow(adata)



print('异常率',ab/total)
print('检出异常:',ab,'样本总数:',total)
print('检测错误数:',total-ab_c,'检测正确异常，正常数:',realab,realn,'样本正确异常，正常数:',labab,labn)
print('准确率:',realab/ab,'召回率:',realab/labab)



