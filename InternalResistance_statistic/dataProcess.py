import csv
import torch
import glob
import numpy as np
import re
import os
import xlrd
import time

path = '/Users/Desktop/电池数据'
files = glob.glob(path + "/*")
# test = Judge("./formula")
# test.show()
k=0
for file in files:
    ID_no=0
    print(file)
    ID_p=0
    dV_p=0
    dTemp_p=0
    with open(file,'r',encoding = 'gb2312') as f1:
        reader=csv.reader(f1)
        for row in reader:
            if len(row)<42:continue
            if not row[42].isdigit() or not row[38].isdigit():#去除null
                continue
            I = float(row[42])
            ID = row[1]
            if I>0 and I<5000:
                ID_no=ID
            if ID!=ID_p:
                if I<0 or I>0:
                    continue
                ID_p=ID
                row_count=0
                dV_p=list(map(float, row[5:21]))
                dTemp_p=list(map(float, row[34:38]))
                I_p=I
                T = float(row[25])
                C = int(row[38])
                mark=0#结束标识
                W=0#能量
                v_mark=0#跳变电压标识
                continue
            if I<=0:
                continue
            if v_mark==0 and I-I_p>1000:
                dV = list(map(lambda x: x[1] - x[0], zip(dV_p, list(map(float, row[5:21])))))
                #dvtest=row[5:21]
                #print(ID)
                dI = I
                if dI==0:
                    print(I,I_p)
                    time.sleep(100)
                v_mark=1
            row_count+=1
            W+=(I/1000)**2
            if row_count>120 :
                if mark==0:#第一次电流下降
                    mark=1
                    dTemp= list(map(lambda x: x[1]-x[0], zip(dTemp_p, list(map(float, row[34:38])))))
                    '''
                    if dI<2000:#测试用
                        print(ID, 'ID')
                        print(dV,'dV')
                        print(dI,'dI')
                        print(T,'T')
                        print(dTemp,'dTemp')
                        print(C,'C')
                        print(row_count,'row_count')
                        print(I,'I')
                        print(W,'W')
                        print(dataTotal,'dataTotal')
                    mm=0
                    if mm==1:
                    '''
                    R_v = [x*1000 / dI for x in dV]
                    R_t=[x*1000 / W for x in dTemp]
                    dataTotal = [ID] + [C] + [T] + R_v +R_t+[row[43]] # ID，充电次数，环境温度，电压变化，电流，温度变化，能量，'时间'
                    R_total=R_v +R_t
                    p=(np.array(R_total)>0).all()
                    #print(p)
                    if ID_no!=ID and p:
                        k += 1
                        if k % 1000 == 0:
                            print(k)
                        if (R_v[0] < 0):
                            print(ID, 'v<0')
                            print((np.array(R_t+R_v)>0).all)
                        if (R_t[0] < 0):
                            print(ID, 't<0')
                        with open('../Data/dataProcessed_I>5000_new2.csv','a+',newline='') as fd:
                            writer=csv.writer(fd)
                            writer.writerow(dataTotal)
                continue



