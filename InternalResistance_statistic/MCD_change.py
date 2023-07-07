import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pyod.models.mcd import MCD
from scipy import stats
from sklearn.covariance import MinCovDet
outlier_fraction = 0.03
pa=0
for tempture in range(0, 40,4):
    print(tempture)
    u1=[]
    u2=[]
    s1=[]
    s2=[]
    s3=[]
    s4=[]
    th=[]
    for count in range(0, 100,10):
        print(count)
        datax = []
        datay = []
        with open('../Data/dataProcessed_I>5000_new2.csv', 'r') as f1:
            reader = csv.reader(f1)
            for row in reader:
                # print(row[2])
                if float(row[1]) >= count and float(row[1]) <= (count+10) and float(row[2]) >= tempture and float(
                        row[2]) <= (tempture+4):
                    for i in range(3, 19):
                        if float(row[i]) <= -100 or float(row[i]) >= 100 or float(row[19]) < -100 or float(
                                row[19]) > 100:
                            continue
                        datax.append(float(row[i]))
                        if i in range(3, 7):
                            datay.append(float(row[19]))
                        if i in range(7, 11):
                            datay.append(float(row[20]))
                        if (i in range(11, 13)) or (i in range(15, 17)):
                            datay.append(float(row[21]))
                        if (i in range(13, 15)) or (i in range(17, 19)):
                            datay.append(float(row[22]))
        # print(datax)
        # print(datay)
        x = np.array(datax)
        y = np.array(datay)
        xy = np.vstack([x, y])  # 将两个维度的数据进行叠加
        est = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None, random_state=None)

        try:
            est.fit(xy.transpose(), None)
            u = est.location_
            s = est.covariance_
            dis = est.mahalanobis(xy.transpose())
            threshold = stats.scoreatpercentile(dis, 100 * (1 - outlier_fraction))
            u1.append(u[0])
            u2.append(u[1])
            s1.append(s[0][0])
            s2.append(s[0][1])
            s3.append(s[1][0])
            s4.append(s[1][1])
            th.append(threshold)
        except:
            u1.append(0)
            u2.append(0)
            s1.append(0)
            s2.append(0)
            s3.append(0)
            s4.append(0)
            th.append(0)
            pa += 1
            print('pass:', pa)
    with open('../Data/u1.csv', 'a+', newline='') as fu1:
        writer = csv.writer(fu1)
        writer.writerow(u1)
    with open('../Data/u2.csv', 'a+', newline='') as fu2:
        writer = csv.writer(fu2)
        writer.writerow(u2)
    with open('../Data/s1.csv', 'a+', newline='') as fs1:
        writer = csv.writer(fs1)
        writer.writerow(s1)
    with open('../Data/s2.csv', 'a+', newline='') as fs2:
        writer = csv.writer(fs2)
        writer.writerow(s2)
    with open('../Data/s3.csv', 'a+', newline='') as fs3:
        writer = csv.writer(fs3)
        writer.writerow(s3)
    with open('../Data/s4.csv', 'a+', newline='') as fs4:
        writer = csv.writer(fs4)
        writer.writerow(s4)
    with open('../Data/th.csv', 'a+', newline='') as fth:
        writer = csv.writer(fth)
        writer.writerow(th)


