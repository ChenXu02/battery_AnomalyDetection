import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import time
np.random.seed(0)
# set dimension of the data
dimx = 10
dimy=10
# create random data, which will be the target values
c1000=0
Z = np.loadtxt("../Data/th.csv",delimiter=",")#更换u1,u2,s1,s2,s3,s4
for i in range(0,len(Z)):
    for j in range(0,len(Z[0])):
        if Z[i][j]>100 or Z[i][j]<=10:
            c1000+=1
            Z[i][j]=0
print(Z.shape,c1000,'zshape')
# create a 2D-mesh
x = np.arange(1,dimx+1).reshape(dimx,1)
y = np.arange(1,dimy+1).reshape(1,dimy)
print(x.shape,y.shape)
X,Y = np.meshgrid(y,x)
print(X.shape,Y.shape)
# calculate polynomial features based on the input mesh
features = {}
features['x^0*y^0'] = np.matmul(x**0,y**0).flatten()
features['x*y^0'] = np.matmul(x,y**0).flatten()
features['x^0*y'] = np.matmul(x**0,y).flatten()
features['x^2*y^0'] = np.matmul(x**2,y**0).flatten()
features['x*y'] = np.matmul(x,y).flatten()
features['x^0*y^2'] = np.matmul(x**0,y**2).flatten()
dataset = pd.DataFrame(features)
print(dataset,'dataset')
# fit a linear regression model
Z_n=Z.flatten()
dataoriginal=dataset
for dz in range(len(Z.flatten())-1,-1,-1):
    if Z.flatten()[dz]==0:
        Z_n=np.delete(Z_n,dz)
        dataset=dataset.drop([dz])
print(dataset.values.shape,Z_n.shape)
reg = LinearRegression().fit(dataset.values, Z_n)
print(reg.coef_,'coe')
print(reg.intercept_,'inte')
cs_re=[]
for i in range(0,len(reg.coef_)):
    cs_re.append(reg.coef_[i])
cs_re.append(reg.intercept_)
with open('../Data/discrete_cs.csv', 'a+', newline='') as fs2:
    writer = csv.writer(fs2)
    writer.writerow(cs_re)
print('ok')
# get coefficients and calculate the predictions
z_pred = reg.intercept_ + np.matmul(dataoriginal.values, reg.coef_.reshape(-1,1)).reshape(dimx,dimy)
cmape=0
mape=0
for i in range(0,len(z_pred)):
    for j in range(0,len(z_pred[0])):
        if Z[i][j]!=0 :
            mape+=(abs(Z[i][j]-z_pred[i][j]))/Z[i][j]
            cmape+=1
print('mape:',mape/cmape)
# visualize the results
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig)
# plot the fitted curve
ax.plot_wireframe(X, Y, z_pred, label = 'prediction')
# plot the target values
ax.scatter(X, Y, Z, c = 'r', label = 'datapoints')
ax.view_init(25, 80)
plt.legend()
plt.show()
