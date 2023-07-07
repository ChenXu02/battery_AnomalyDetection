import csv
from numpy import *
import torch
import glob
import numpy as np
import re
import os
import xlrd
import time
atime=time.time()
def SG(T,C,R):#C,T
    x=int(C/10)
    y=int(T/4)
    cs = np.loadtxt("../Data/discrete_cs.csv", delimiter=",")
    features = np.zeros((1,6))
    features[0][0] = (x**0* y ** 0)
    features[0][1] = (x * y**0)
    features[0][2] = (x ** 0* y)
    features[0][3] = (x ** 2* y ** 0)
    features[0][4] = (x* y)
    features[0][5] = (x ** 0* y ** 2)
    u1 = np.squeeze(np.matmul(features, cs[0][0:6].reshape(-1, 1))+cs[0][-1])
    u2 = np.squeeze(np.matmul(features, cs[1][0:6].reshape(-1, 1)) + cs[1][-1])
    th = np.squeeze(np.matmul(features, cs[6][0:6].reshape(-1, 1)) + cs[6][-1])
    s=[[0.1*abs(R)/15,0],[0,20]]
    u=[u1,u2]
    return u,s,th
def get_mahalanobis(D, u, x):
    invD = np.linalg.inv(D)  # 协方差逆矩阵
    tp = np.array(x)-np.array(u)
    return np.sqrt(dot(dot(tp, invD), tp.T))
path = '/Users/Desktop/验证数据/数据'
files = glob.glob(path + "/*")
# test = Judge("./formula")
# test.show()
k=0
chargeTotal=0
abtotal=0
nototal=0
aandoTotal=0
dianxincn=0
dianxinca=0
abtotall=0
nototall=0
aandoTotall=0
dianxincnl=0
dianxincal=0
dianxinTotal=0
datam_p=[]
for file in files:
    ID_no=0
    print('file name:',file)
    ID_p = 0
    dV_p = 0
    dTemp_p = 0
    with open(file,'r',encoding = 'gb2312') as f1:
        reader=csv.reader(f1)
        for row in reader:
            try:
                if not row[42].isdigit() or not row[38].isdigit():#去除null
                    continue
            except:
                continue
            I = float(row[42])
            ID = row[39]
            if ID!=ID_p:
                if I!=0:
                    continue
                ID_p = 0
                dV_p = 0
                dTemp_p = 0
                if I<0 or int(row[3])==0:
                    continue
                ID_p=ID
                datam = []
                datam.append(row)
                row_count=0
                dV_p=list(map(float, row[5:21]))
                maxt=[float(row[34]),float(row[35]),float(row[36]),float(row[37])]
                for i in range(0, 4):
                    if maxt[i]==0:
                        maxt[i]=max(maxt)
                dTemp_p=maxt
                I_p=I
                T = float(row[25])
                C = int(row[38])
                mark=0#结束标识
                W=0#能量
                v_mark=0#跳变电压标识
                continue
            if I<=0:
                continue
            if v_mark==0 and (I-I_p>1000 or I>5000):
                dV = list(map(lambda x: x[1] - x[0], zip(dV_p, list(map(float, row[5:21])))))
                dI = I
                v_mark=1
            datam.append(row)
            row_count+=1
            W+=(I/1000)**2
            if row_count>120 :
                if v_mark==0:
                    continue
                if mark==0:#第一次电流下降
                    mark=1
                    dTemp= list(map(lambda x: x[1]-x[0], zip(dTemp_p, list(map(float, row[34:38])))))
                    dTemp=[dTemp[0],dTemp[0],dTemp[0],dTemp[0],dTemp[1],dTemp[1],dTemp[1],dTemp[1],dTemp[2],dTemp[2],dTemp[3],dTemp[3],dTemp[2],dTemp[2],dTemp[3],dTemp[3]]
                    R_v = [x*1000 / dI for x in dV]
                    R_t=[x*1000 / W for x in dTemp]
                    u1 = sum(sorted(R_v)[6:10]) / 4
                    u2= sum(sorted(R_t)[6:10]) / 4
                    if u1<=0 or u2<=0:
                        continue
                    dataTotal = [ID] + [C] + [T] + R_v +R_t+[row[43]] # ID，充电次数，环境温度，电压变化，电流，温度变化，能量，'时间'
                    #print(p)
                    uo,s,t0=SG(C,T,u1)
                    u = [u1, u2]
                    t1=t0*0.07+11
                    t2=(t0*0.07+15)*2
                    #print(t)
                    dianxinmarkn=0
                    dianxinmarka = 0
                    dianxinmarkln = 0
                    dianxinmarkla=0
                    dianxinc=0
                    datam_pp=[]
                    chargeTotal += 1
                    for i in range(0,len(R_v)):#每个电芯
                        dianxinTotal+=1
                        P=[R_v[i],1+R_t[i]*0.1]

                        dis=get_mahalanobis(s,u,P)
                        #print(t)
                        adata = [ID] + [C] + [T] + [R_v[i]] + [R_t[i]] + [row[43]] + [i] + [t1]+ [t2] + [dis] + [' ']
                        if(dis>=t1 and dis<t2):
                            if (R_v[i]<0):
                                adata = adata + ['abnormal']
                                dianxinca += 1
                                dianxinmarka = 1
                            else:
                                adata = adata+['notice']
                                dianxincn += 1
                                dianxinmarkn=1
                        if ( dis >= t2 ):
                            dianxinca += 1
                            adata = adata + ['abnormal']
                            dianxinmarka = 1
                        if (np.abs(R_v[i]-u1) >= np.abs(0.5*u1) and np.abs(R_v[i]-u1) < np.abs(u1)):
                            adata[10] = 'notice_label'
                            dianxincnl+=1
                            dianxinmarkln=1
                        if ( np.abs(R_v[i]-u1) > np.abs(u1)):
                            adata[10] =  'abnormal_label'
                            dianxinmarkla=1
                            dianxincal+=1
                        datam_pp.append(adata)
                            #print('ID:', ID, 'C:', C, 'T:', T, 'dis:', dis, 'threshold:', t)
                    if dianxinmarkla == 1:
                        abtotall += 1
                        aandoTotall += 1

                    if dianxinmarkln == 1:
                        nototall += 1
                        aandoTotall += 1
                    if dianxinmarka == 1:
                        abtotal += 1
                        aandoTotal += 1
                    if dianxinmarkn == 1:
                        nototal += 1
                        aandoTotal += 1
                    if dianxinmarka==1 or dianxinmarkn==1 or dianxinmarkln==1 or dianxinmarkla==1:
                        datam_p = datam_p + datam_pp
                        with open('../Data/validation_t2.csv', 'a+', newline='') as fth:
                            writer = csv.writer(fth)
                            for ii in range (0,len(datam)):
                                writer.writerow(datam[ii])
                continue
for i in range(len(datam_p)):
    with open('../Data/validation_p_t2.csv', 'a+', newline='') as fth2:
        writer = csv.writer(fth2)
        writer.writerow(datam_p[i])
print(abtotal,chargeTotal)
print('non_fixed')
print('电池包总异常率',aandoTotal/chargeTotal)
print('notice电池包总异常率',nototal/chargeTotal)
print('abnormal电池包异常率',abtotal/chargeTotal)
print('notice电芯异常率',dianxincn/dianxinTotal)
print('abnormal电芯异常率',dianxinca/dianxinTotal)
print('fixed')
print('电池包总异常率',aandoTotall/chargeTotal)
print('notice电池包总异常率',nototall/chargeTotal)
print('abnormal电池包异常率',abtotall/chargeTotal)
print('notice电芯异常率',dianxincnl/dianxinTotal)
print('abnormal电芯异常率',dianxincal/dianxinTotal)