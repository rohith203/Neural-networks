import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# if __name__=='__main__':
model = LinearRegression()
# model = SGDRegressor()

# data input
data = pd.read_csv("./3D_spatial_network.csv", header=None)[::]
X = data.loc[:,0:1].values
y = data.loc[:,2].values


# data preprocessing (Normal scaling)

# print("hell")
scaler = StandardScaler()
X = scaler.fit_transform(X)
# scaler.fit(X[:,0].reshape(-1,1),axis=1)
# X[:,0] = scaler.transform(X[:,0].reshape(-1,1)).reshape(-1,)
# scaler.fit(X[:,1].reshape(-1,1))
# X[:,1] = scaler.transform(X[:,1].reshape(-1,1)).reshape(-1,)
# X[:,0] = normalize(X[:,0].reshape(-1,1)).reshape(-1,)

# Training the model by choosing alpha and max_iter values.
# gradient descent algorithm can be set as either ‘batch’ or ‘stochastic’
# in this function call.
alpha = 0.007
max_iter = 200
algo = 'stochastic'
arr = model.fit(X,y)
costf = sum((model.predict(X)-y)**2)
# arr = model.train(X,y,0.19,250,"stochastic")
print("coefficients: ",model.coef_)
print("intercept: ",model.intercept_)
print("Total Cost: ",(0.5/X.shape[0])*costf)



# W_arr = np.array(model.W_arr)
res = 50
bounds = [16,16]
# xx = np.linspace(model.coef_[0]-bounds[0],model.coef_[0]+bounds[0], res)
# yy = np.linspace(model.coef_[1]-bounds[1],model.coef_[1]+bounds[1], res)
xx = np.linspace(-10,13, res)
yy = np.linspace(-10,10, res)
# minw0 = W_arr[-1][0][0]

r = np.ndarray((res,res))
s = np.ndarray((res,res))
z = np.ndarray((res,res))
def get_cost(W):
    b = np.ones((X.shape[0],1))
    h = np.matmul(np.concatenate((b,X), axis=1),W)
    return (0.5/X.shape[0])*sum((h-y.reshape(-1,1))**2)


for i in range(res):
    for j in range(res):
        # w_c = np.array([ [model.intercept_] ,[xx[i]], [yy[j]] ])
        h = 0
        for k in range(X.shape[0]):
            h+=((xx[i]*X[k][0] + yy[j]*X[k][1] - y[k])**2)

        z[i][j] = (h*0.5)/X.shape[0]
        # z[i][j] = get_cost(w_c)
        r[i][j] = xx[i]
        s[i][j] = yy[j]

# 3d surface plot of cost function and learning curve
ax = plt.axes(projection='3d')
ax.plot_surface(r, s, z,cmap='coolwarm')
# ax.plot(W_arr[:,1], W_arr[:,2], model.cost_arr,c='red')
ax.text2D(0.05, 0.95, "3D surface plot of cost function ({2})\n alpha={0} max_iter={1}".format(alpha,max_iter,algo), transform=ax.transAxes)
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("cost")
# plt.savefig("./Results/lin_reg/{2}_{0}_{1}_surf.png".format(alpha,max_iter,algo))
plt.show()


plt.figure()
plt.title("2d contour plot of cost function ({2})\n alpha={0} max_iter={1}".format(alpha,max_iter,algo))
plt.xlabel("w1")
plt.ylabel("w2")
plt.contour(r,s,z.reshape(res,res),levels=25)
plt.show()