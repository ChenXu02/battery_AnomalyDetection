import os
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn import linear_model
from scipy import signal
from scipy.stats import gaussian_kde
from scipy import stats
from sklearn.covariance import MinCovDet
from numpy import *
def get_mahalanobis(D, u, x):

    invD = np.linalg.inv(D)  # 协方差逆矩阵
    tp = np.array(x)-np.array(u)
    return dot(dot(tp, invD), tp.T)

def colarShow(x,y,x2,y2,I,T):
    x = np.array(x)
    y = np.array(y)
    xy = np.vstack([x, y])  # 将两个维度的数据进行叠加

    # print(xy)
    kenal = gaussian_kde(xy)  # 这一步根据xy这个样本数据，在全定义域上建立了概率密度分布，所以kenal其实就是一个概率密度函数，输入对应的(x,y)坐标，就给出相应的概率密度
    # kenal.set_bandwidth(bw_method=kenal1.factor)#这一步可以直接设置bandwith，通常情况下默认即可
    z = kenal.evaluate(xy)  # 得到我们每个样本点的概率密度
    # z = gaussian_kde(xy)(xy)  # 这行代码和上面两行是相同的意思，这行是一行的写法
    idx = z.argsort()  # 对z值进行从小到大排序并返回索引
    x, y, z = x[idx], y[idx], z[idx]  # 对x,y按照z的升序进行排列
    # 上面两行代码是为了使得z值越高的点，画在上面，不被那些z值低的点挡住，从美观的角度来说还是十分必要的

    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    ax.grid()
    scatter = ax.scatter(x, y, marker='o', c=z, edgecolors='none', s=25, label='label'
                         , cmap='Spectral_r')
    cbar_ax = plt.gcf().add_axes([0.93, 0.15, 0.02, 0.7])  # [left,bottom,width,height] position
    cbar = fig.colorbar(scatter, cax=cbar_ax, label='Probability density')

    titles = "SOC 50,100"
    ax.set_title(titles)

    #ax.plot(x2, y2)
    #plt.show()
    outlier_fraction = 0.05
    est = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None, random_state=None)
    est.fit(xy.transpose(), None)
    u = est.location_
    s = est.covariance_
    s[0][0]*=1.1
    s[1][1] *= 0.9


    print(u,s,'usus',xy.transpose()[0])
    #dis=[]
    #for i in range(0,len(xy.transpose())):
        #dis.append(get_mahalanobis(s,u,xy.transpose()[i]))
    dis=est.mahalanobis(xy.transpose())



    threshold = stats.scoreatpercentile(dis, 100 * (1 - outlier_fraction))

    newdata = []
    total=0
    norm=0
    for i in range(len(dis)):
        if dis[i]<threshold:
            norm+=1
            newdata.append(xy.transpose()[i])
    print('异常：',(len(dis)-norm)/len(dis),len(dis))
    n_data=np.array(newdata).T
    #func3 = np.polyfit(n_data[0], n_data[1], 1)
    #yn3 = np.poly1d(func3)(x2)
    #ax.plot(x2, yn3,'red')
    xx, yy = np.meshgrid(np.linspace(min(x2), max(x2), 100), np.linspace(max(y), -5, 1000))
    plt.figtext(0.4, 0.2, f"I:%s, "% I+f"T:%s, "% T+"th: {:.2f}"  .format(threshold), fontsize='16')
    #Z = est.mahalanobis(np.c_[xx.ravel(), yy.ravel()])
    Z=[]
    for i in range(0,len(np.c_[xx.ravel(), yy.ravel()])):
        Z.append(get_mahalanobis(s,u,np.c_[xx.ravel(), yy.ravel()][i]))
    Z = np.array(Z).reshape(xx.shape)
    # print(dis,'dis')
    # print(len(Z),len(Z[0]),Z)
    ax.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    plt.show()
    #print('u:', u)
    return u,s,threshold
def show(V_I,ID,std,savename):
    V= np.transpose(V_I)#转置
    num_plots = 2
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
    # Plot several different functions..
    labels = []
    x_axis=range(0,len(V[0]),1)
    for i in range(0, num_plots ):
        x, y = x_axis, V[i]  # 接收参数
    #x, y = range(0, len(V[0]), 1), V[i]
        plt.plot(x, y)
        labels.append(r'%i ' %  (i*50))
    plt.title(' ', fontsize='16')  # 设置图的名称
    plt.xlabel('I (2000+I*1000)', fontsize='12')  # 设置x轴的名称,可调字的大小等
    plt.ylabel('u', fontsize='12')  # 设置y轴的名称，可调字的大小等
    plt.ylim(0, 5)


    plt.figtext(0.3, 0.3, f"T:%s" %ID,fontsize='16')
    #plt.figtext(0.3, 0.5, f"STD:%s" % std, fontsize='16')

    plt.legend(labels, ncol=4, loc='lower right',
           bbox_to_anchor=[1, 0],
           columnspacing=1.0, labelspacing=0.2,
           handletextpad=0.2, handlelength=1.5,
           fancybox=True, shadow=True)

    #plt.savefig('Results/{}_{}.jpg' .format(ID, savename))  #

    plt.show()

def V_extraction(path,chargeType):
    ID_p = 0  # 上一个电池的ID
    STD_record = []  # 记录电压
    I_record = []  # 记录电流
    T_record=[]
    SOC0 = -1
    SOC50 = -1
    SOC70 = -1

    for file_name in os.listdir(path):
        print(file_name)
        if ".csv" not in file_name:
            continue
        open_file = path + file_name
        with open(open_file, 'r', encoding='gb18030') as f1:
            reader = csv.reader(f1)
            for row in reader:
                if len(row) < 42: continue
                ID = row[39]
                if ID[0:5] != chargeType: continue
                if not row[42].isdigit() or not row[38].isdigit():  # 去除null
                    continue
                I = float(row[42])
                T = float(row[25])  # 环境温度
                SOC =float(row[4])

                if ID != ID_p:  # 到达新的ID
                    ID_p = ID
                    SOC0 = -1
                    SOC50 = -1
                    SOC70 = -1
                    T0 = T
                    if I >= 0 and SOC0 == -1:
                        SOC0 = np.std(list(map(float, row[5:21])))

                if SOC==50 and SOC50==-1:
                    SOC50=np.std(list(map(float, row[5:21])))

                    T_record.append(T0)
                    I_record.append(I)

                if SOC==70 and SOC70==-1:
                    SOC70=1
                    if SOC50!=-1:
                        SOC50=(SOC50+np.std(list(map(float, row[5:21]))))/2
                        STD_record.append([SOC0,SOC50])
                    else:
                        SOC50 =np.std(list(map(float, row[5:21])))
                        STD_record.append([SOC0, SOC50])

    np.savez('STD_BAAC9.npz', STD=STD_record, I=I_record,T=T_record)
def dataP():
    datas = np.load('STD_BAAC9.npz')
    STD_record=datas['STD']
    data_show=STD_record
    x_curver = np.linspace(-10, 20, 100)
    #print(SOH_STD[:,1].shape, SOH_STD[:,0].shape)
    STD_record_n=np.array(data_show)
    SOC50_show=list(STD_record_n[:,0])
    SOC100_show=list(STD_record_n[:,1])
    funcr1 = np.polyfit(SOC50_show,SOC100_show , 1)
    yr1 = np.poly1d(funcr1)
    u,s,th=colarShow(SOC50_show, SOC100_show, x_curver, yr1(x_curver), ' {:.2f}'.format(1), ' {:.2f}'.format(1))
    print(u,s,th)
    #np.savez('paraSave_BAAC9_0.05.npz', u=u, s=s, th=th)

path = '/Users/didi/Desktop/电池数据/SOCtrainData/'
#path = '/Users/didi/Desktop/电池数据/BCB99/BCB99/'
#'BLEC9'#'BHHCe'#'BCBC9'#'BCB8c'#'BBEC9'#'BCB99'#'BAAC9'#'BJJCf'
chargeType='BAAC9'#'BCB99'
#V_extraction(path,chargeType)
dataP()


