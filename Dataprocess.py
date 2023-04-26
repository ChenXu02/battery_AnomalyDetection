import os
import csv
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from ICA import ICA,moving_average,Diff
from sklearn import linear_model
from scipy import signal
from scipy.stats import gaussian_kde
from scipy import stats
from sklearn.covariance import MinCovDet
plt.rcParams['font.sans-serif']=['SimHei']#支持图中显示汉字
def AxisTrans(x,y):
    re=np.concatenate([x, y], axis=0).reshape(2,-1)#电压和Q合并，转换成两行数据
    re=re.T[np.argsort(re.T[:, 0])].T
    Q_record=-100
    re_new=[]
    for i in range(len(re[0])-1):
        if re[1][i]>=Q_record:
            if re[0][i]!=re[0][i+1]:
                Q_record=re[1][i]
                re_new.append(re[:,i])
        else:
            continue
    re_new=np.array(re_new).T
    return re[0],re[1],re_new[0],moving_average(re_new[1],3,0)
def show(V,I,ID,mark,V_x):
    V= np.transpose(V)#转置
    num_plots = 16
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
    # Plot several different functions..
    labels = []
    for i in range(0, num_plots ):
        x, y = V_x[i], V[i]  # 接收参数
        #x, y = range(0, len(V[0]), 1), V[i]
        #plt.plot(x, y)
        labels.append(r'%i ' %  i)
        plt.title('Voltage-Time', fontsize='16')  # 设置图的名称
        plt.xlabel('Time', fontsize='12')  # 设置x轴的名称,可调字的大小等
        plt.ylabel('Voltage', fontsize='12')  # 设置y轴的名称，可调字的大小等
    #plt.ylim(-20, 20)
    plt.twinx()#设置第二个坐标轴
    plt.plot(V_x[0], I)
    #plt.plot(range(0, len(V[0]), 1), I)
    plt.ylabel('I', fontsize=9)
    #plt.ylim(-20, 20)
    labels.append(r'I ')

    plt.figtext(0.3, 0.3, f"SOC:%s" %ID,fontsize='16')

    plt.legend(labels, ncol=4, loc='lower right',
           bbox_to_anchor=[1, 0],
           columnspacing=1.0, labelspacing=0.2,
           handletextpad=0.2, handlelength=1.5,
           fancybox=True, shadow=True)
    if mark==0:
        plt.savefig('Results/%s_original.jpg' % ID)  #
    else:
        plt.savefig('Results/%s.jpg' % ID)  #
    plt.show()
def simpleshow(x,y,SOC,mark=1):
    if mark:
        plt.plot(x, y)
    else:
        plt.scatter(x, y)
    plt.figtext(0.3, 0.3, f"SOC:%s" % SOC, fontsize='16')
    plt.show()
def colarShow(x,y,x2,y2,SOC):
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
    titles = "SOC-dQ/dV"
    ax.set_title(titles)
    ax.plot(x2, y2)
    #plt.show()
    est = MinCovDet(store_precision=True, assume_centered=False, support_fraction=None, random_state=None)
    est.fit(xy.transpose(), None)
    u = est.location_
    s = est.covariance_
    dis = est.mahalanobis(xy.transpose())
    outlier_fraction = 0.3
    threshold = stats.scoreatpercentile(dis, 100 * (1 - outlier_fraction))
    newdata = []
    for i in range(len(dis)):
        if dis[i]<threshold:
            newdata.append(xy.transpose()[i])
    n_data=np.array(newdata).T
    func3 = np.polyfit(n_data[0], n_data[1], 1)
    yn3 = np.poly1d(func3)(x2)
    ax.plot(x2, yn3,'red')
    xx, yy = np.meshgrid(np.linspace(min(x2), max(x2), 100), np.linspace(max(y), min(y), 1000))
    plt.figtext(0.2, 0.2, f"SOC:%s, "% SOC+" {:.2f}"  .format(func3[0]), fontsize='16')
    Z = est.mahalanobis(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # print(dis,'dis')
    # print(len(Z),len(Z[0]),Z)
    ax.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
    plt.show()
    #print('u:', u)
def regress(M, x, x_n, t_n, lamda=0):
    N=len(x_n)
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    e = np.tile(order, [1,N])
    XT = np.power(x_n, e)
    X = np.transpose(XT)
    a = np.matmul(XT, X) + lamda*np.identity(M+1) #X.T * X
    b = np.matmul(XT, t_n) #X.T * T
    w = np.linalg.solve(a,b) #aW = b => (X.T * X) * W = X.T * T
    print("W:")
    print(w)
    e2 = np.tile(order, [1,x.shape[0]])
    XT2 = np.power(x, e2)
    p = np.matmul(w, XT2)
    return p
def record(data):
    with open('SOCtrainData.csv','a+',newline=' ') as f0:
        writer=csv.writer(f0)
        writer.writerow(data)
path = 'Desktop/电池数据/SOCtrainData2/BJJCf/'
path_ID = 'Desktop/电池数据/SOCtrainData2/'
chargeType='***'
maxV_r1 = []  # 记录最大值
maxV_p1 = []  # 记录最大值位置
maxV_r2 = []  # 记录最大值
maxV_p2 = []  # 记录最大值位置
maxV_s1 = []  # 记录最大值
maxV_s2_0 = []  # 记录最大值
maxV_s2_1 = []  # 记录最大值
maxV_s2_2 = []  # 记录最大值
maxV_s2 = []  # 记录最大值
V_SOC = []
for i in range(0,12):#选取不同SOC段的数据
    b=500*i
    ID_p = 0#上一个电池的ID
    C_p=0#上一个电池的充电次数
    T_p=0#上一个电池的温度
    V_record = []#记录电压
    I_record = []#记录电流
    c=0
    count=0
    ID_record=[]#记录ID
    data1=[]
    data2=[]
    label=[]
    meanQV=[]
    meanSOC=[]
    for file_name in os.listdir(path):
        print(file_name)
        if ".csv" not in file_name:
            continue
        open_file=path+file_name
        SOCID_file=path_ID+'ID/SOCID.csv'
        with open(open_file,'r',encoding='gb18030') as f1:
            reader=csv.reader(f1)
            mark=0
            for row in reader:
                if len(row)<42:continue
                if not row[42].isdigit() or not row[38].isdigit():#去除null
                    continue
                I = float(row[42])
                ID = row[39]
                if ID[0:5]!=chargeType:continue
                T = float(row[25])  # 环境温度
                C = int(row[38])  # 充电次数
                if ID_p==0:
                    ID_p=ID
                    C_p=C
                    T_p=T
                if ID!=ID_p :#到达新的ID
                    SOC=0
                    if len(V_record)>=10:

                        with open(SOCID_file,'r',encoding='gb18030') as f2:#查询该电池的SOC
                            reader2=csv.reader(f2)
                            for row2 in reader2:
                                if row2[0]==ID_p:
                                    SOC=row2[1]
                                    break
                        pos=int(np.argmax(np.array(V_record)[:,0:16])/16)#寻找最大值索引
                        if pos>10:
                            V_record = np.array(V_record[0:pos])
                            V_record_1=np.mean(V_record[:,0:16],axis=0)#求均值，找最大值
                            maxp=np.argmax(V_record_1)
                            #print(maxp)
                            #print(V_record.shape,'vvvvvsss')
                            I_record=I_record[0:pos]
                            dx=maxp
                            if float(SOC)>=21600+b and float(SOC)<25200+b and T_p>=26 and T_p<=27:#温度大于25度
                                ID_record.append(ID_p)
                                #data1.append(V_record[:,dx])
                                #data2.append([T_p,C_p])
                                #label.append(SOC)
                                _,Q=ICA(V_record[:,dx],I_record)
                                V_o,Q_o,V_n,Q_n=AxisTrans(V_record[:,dx],Q)
                                '''
                                #函数拟合代码
                                x_curve = np.linspace(3300, 3400, 300)
                                func = np.polyfit(V_n, Q_n, 50)
                                yn = np.poly1d(func)
                                '''
                                try:
                                    SOC_n=float(SOC)
                                    func = interp1d(V_n, Q_n, kind='cubic')  # 插值拟合
                                    x_curve = np.linspace(start=3300, stop=3400, num=1000)  # 在start和stop间拟合num个数据
                                    yn = func(x_curve)
                                    simpleshow(x_curve, moving_average(Diff(yn)/Diff(x_curve),50,0),SOC_n)#画图
                                    data_qv_smooth=moving_average(Diff(yn) / Diff(x_curve), 1, 0)
                                    r1=max(data_qv_smooth[0:700])
                                    r2=max(data_qv_smooth[700:1000])
                                    p1=np.argmax(data_qv_smooth[0:700])
                                    p2=700+np.argmax(data_qv_smooth[700:1000])
                                    s1_data=data_qv_smooth[p1-300:p1]
                                    s1 = np.sum(s1_data[s1_data>0])
                                    s2_0_data=data_qv_smooth[p1:1000]
                                    s2_0 = np.sum(s2_0_data[s2_0_data>0])
                                    s2_1_data=data_qv_smooth[p1:1000]-(r1-0.025)
                                    s2_1=np.sum(s2_1_data[s2_1_data>0])
                                    s2_2_data=data_qv_smooth[p1:p2]-(r1-0.025)
                                    s2_2=np.sum(s2_2_data[s2_2_data>0])
                                    s2_data = data_qv_smooth[p1:p2]
                                    s2=np.sum(s2_data[s2_data>0])
                                    #print(np.argmax(data_qv_smooth[0:650]),650+np.argmax(data_qv_smooth[650:1000]))
                                    #s1=np.sum(data_qv_smooth[np.argmax(data_qv_smooth[0:650])-30:np.argmax(data_qv_smooth[0:650])])
                                    meanQV.append(data_qv_smooth)
                                    meanSOC.append(SOC_n)
                                    maxV_r1.append(r1)
                                    maxV_p1.append(p1)
                                    maxV_r2.append(r2)
                                    maxV_p2.append(p2)
                                    maxV_s1.append(s1)
                                    maxV_s2_0.append(s2_0)
                                    maxV_s2_1.append(s2_1)
                                    maxV_s2_2.append(s2_2)
                                    maxV_s2.append(s2)
                                    V_SOC.append(SOC_n)
                                except:
                                    pass
                    V_record=[]
                    I_record=[]
                    ID_p=ID
                    C_p = C
                    T_p = T
                    count = 0
                dV_p=list(map(float, row[5:21]))+[float(row[42])]#电压数据
                V_record.append(dV_p)
                I_record.append(float(row[42]))
                count+=1
            x_curver1 = np.linspace(0, 0.1, 10)
            x_curver2 = np.linspace(0, 0.1, 10)
            x_curvep1 = np.linspace(200, 700, 100)
            x_curvep2 = np.linspace(700, 1000, 100)
            x_curves2 = np.linspace(6, 12, 11)#p1-1000
            x_curves1 = np.linspace(4, 9, 100) #300
            x_curves2_0 = np.linspace(7, 15, 100)#p1-p2
            x_curves2_1 = np.linspace(0, 2, 100)#p1-1000 -base
            x_curves2_2 = np.linspace(0, 2, 100)#p1-p2 -base
            x_curver1r2 = np.linspace(0, 0.05, 10)
            x_curvep1p2 = np.linspace(200, 500, 100)
            x_curve2 = np.linspace(0, 0.1, 10)
            x_curveC = np.linspace(-0.6, 0.6, 10)
            x_curvepC = np.linspace(0, 600, 100)
            funcr1 = np.polyfit(maxV_r1, V_SOC, 1)
            yr1 = np.poly1d(funcr1)
            funcr2 = np.polyfit(maxV_r2, V_SOC, 1)
            yr2 = np.poly1d(funcr2)
            funcp1 = np.polyfit(maxV_p1, V_SOC, 1)
            yp1 = np.poly1d(funcp1)
            funcp2 = np.polyfit(maxV_p2, V_SOC, 1)
            yp2 = np.poly1d(funcp2)
            funcs1 = np.polyfit(maxV_s1, V_SOC, 1)
            ys1 = np.poly1d(funcs1)
            funcs2_0 = np.polyfit(maxV_s2_0, V_SOC, 1)
            ys2_0 = np.poly1d(funcs2_0)
            funcs2_1 = np.polyfit(maxV_s2_1, V_SOC, 1)
            ys2_1 = np.poly1d(funcs2_1)
            funcs2_2 = np.polyfit(maxV_s2_2, V_SOC, 1)
            ys2_2 = np.poly1d(funcs2_2)
            funcs2 = np.polyfit(maxV_s2, V_SOC, 1)
            ys2 = np.poly1d(funcs2)
#####################################################
            maxV_C=(np.array(maxV_r1)-np.array(maxV_r2))
            maxV_pC = (np.array(maxV_p2) - np.array(maxV_p1))
            funcC = np.polyfit(maxV_C, V_SOC, 1)
            funcpC = np.polyfit(maxV_pC, V_SOC, 1)
            yC = np.poly1d(funcC)
            ypC = np.poly1d(funcpC)
            #maxV_sC=
            colarShow(maxV_r1, V_SOC, x_curver1, yr1(x_curver1), 'r1, {:.2f}'.format(funcr1[0]))
            colarShow(maxV_r2, V_SOC, x_curver2, yr2(x_curver2), 'r2, {:.2f}'.format(funcr2[0]))
            colarShow(maxV_p1, V_SOC, x_curvep1, yp1(x_curvep1), 'p1, {:.2f}'.format(funcp1[0]))
            colarShow(maxV_p2, V_SOC, x_curvep2, yp2(x_curvep2), 'p2, {:.2f}'.format(funcp2[0]))
            colarShow(maxV_s1, V_SOC, x_curves1, ys1(x_curves1), 's1, {:.2f}'.format(funcs1[0]))
            colarShow(maxV_s2_0, V_SOC, x_curves2_0, ys2_0(x_curves2_0), 's2_0, {:.2f}'.format(funcs2_0[0]))
            colarShow(maxV_s2, V_SOC, x_curves2, ys2(x_curves2), 's2, {:.2f}'.format(funcs2[0]))
            colarShow(maxV_s2_1, V_SOC, x_curves2_1, ys2_0(x_curves2_1), 's2_1, {:.2f}'.format(funcs2_1[0]))
            colarShow(maxV_s2_2, V_SOC, x_curves2_2, ys2_0(x_curves2_2), 's2_2, {:.2f}'.format(funcs2_2[0]))
            colarShow(maxV_C, V_SOC, x_curver1r2, yC(x_curver1r2), 'r1-r2, {:.2f}' .format(funcC[0]))
            colarShow(maxV_pC, V_SOC, x_curvep1p2, ypC(x_curvep1p2), 'p2-p1,{:.2f}'.format(funcpC[0]))
            #colarShow(maxV_r1, V_SOC, x_curver1, yr1(x_curver1), 'Correlation r1')
            #colarShow(maxV_C, V_SOC,x_curveC,yC(x_curveC), 'Correlation test rC0')
            #colarShow(maxV_r2, V_SOC,x_curve2,yn2(x_curve2), 'Correlation test r2')
            #colarShow(maxV_p1, V_SOC, 'Correlation test p1')
            #colarShow(maxV_p2, V_SOC, 'Correlation test p2')
            #colarShow(maxV_s1, V_SOC, 'Correlation test s1')
            print(len(maxV_s2),len(V_SOC))
            #colarShow(maxV_s2, V_SOC,x_curve,yp, 'Correlation test s2(p1-p2)')
            #colarShow(maxV_s2, maxV_r2, x_curve, ynsr(x_curve), 'Correlation test nsr')
            # 函数拟合代码
        meanQV_show=np.mean(np.array(meanQV),axis=0)
        meanSOC_show="{:.2f}".format(np.mean(np.array(meanSOC)))
        x_curve = np.linspace(start=3300, stop=3400, num=1000)  # 在start和stop间拟合num个数据
        #simpleshow(x_curve, meanQV_show, meanSOC_show)
    '''
    #存储数据
    Trainfilename='SOCtrainData_'+chargeType+'_posTop30_{}.npz' .format(b)
    np.savez(Trainfilename, ID=ID_record,VandI = data1, CandT = data2,label=label)
    rdata=np.load(Trainfilename)
    print(rdata.files)
    print(rdata['ID'].shape)
    print(rdata['label'].shape)
    print(rdata['VandI'].shape)
    print(rdata['CandT'].shape)
    '''

