import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    
    def init_weights(self, s):
        np.random.seed(11)
        self.W = np.random.randn(s[1],1)
        self.W_arr = []
        self.cost_arr = []
        self.cost = 0
        self.gradient = 0
        
    def get_cost(self, X, y, W):
        total_cost = 0
        for i in range(X.shape[0]):
            total_cost += np.square(self.get_h_i(X, i, W)-y[i])
        eta = 0.1
        reg = 0
        for i in range(1,W.shape[0]):
            reg += abs(W[i][0])
        return (0.5/X.shape[0])*total_cost + 0.5*eta*reg
    
    def add_bias(self, X):
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X), axis=1)
    
    def get_h_i(self, X, i, W):
        h_i = 0
        for j in range(X.shape[1]):
            h_i += X[i][j]*W[j][0]
        return h_i
    
    def batch_grad_descent(self, X, y, alpha, eta, max_iter):
        for iteration in range(max_iter):
            W_new = np.ndarray(self.W.shape)
            for j in range(X.shape[1]):
                grad = 0
                for i in range(X.shape[0]):
                    grad += (self.get_h_i(X, i, self.W) - y[i])*X[i][j]
                W_new[j][0] = self.W[j][0]-0.5*eta*alpha*np.sign(self.W[j][0]) - (alpha/X.shape[0])*grad
            self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
        return W_new
    
    def stochastic_grad_descent(self, X, y, alpha, eta, max_iter):
        for iteration in range(max_iter):
            W_new = self.W.copy()
            for i in range(X.shape[0]):    
                ind = np.random.randint(X.shape[0])
                grad = 0
                for j in range(X.shape[1]):
                    grad += (self.get_h_i(X, ind, self.W) - y[ind]) * X[ind][j]
                for j in range(X.shape[1]):
                    W_new[j][0] = self.W[j][0]-0.5*eta*alpha*np.sign(self.W[j][0]) - (alpha/X.shape[0])*grad
                self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
        return self.W
                              
        
    def train(self, X, y, alpha, eta, max_iter=100, option="batch"):
        X = self.add_bias(X)
        self.init_weights(X.shape)
        if option=="batch":
            self.batch_grad_descent(X,y,alpha,eta,max_iter)
        elif option=="stochastic":
            self.stochastic_grad_descent(X,y,alpha,eta,max_iter)
        self.cost = self.cost_arr[len(self.cost_arr)-1]
        return self.cost_arr
        
    def test(self,X):
        X = self.add_bias(X)
        y_pred = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                y_pred[i] += X[i][j]*self.W[j][0]
        return y_pred

class NormalScaler:
    
    def fit(self, arr):
        self.mean = np.mean(arr)
        self.std = np.std(arr)
        
    def transform(self, arr):
        return (arr-self.mean)/(self.std)
        

class MinMaxScaler:
    
    def fit(self, arr):
        self.min = np.min(arr)
        self.max = np.max(arr)
        
    def transform(self, arr):
        return (arr-self.min)/(self.max-self.min)

    def inv_transform(self, arr):
        return arr*(self.max-self.min)+self.min

if __name__ == "__main__":
    model = LinearRegression()

    data = pd.read_csv("./data.csv", header=None)
    # print(data)
    X = data.loc[:,0:1].values
    y = data.loc[:,2].values
    mscaler = MinMaxScaler()
    mscaler.fit(X[:,0])
    X[:,0] = mscaler.transform(X[:,0])
    mscaler.fit(X[:,1])
    X[:,1] = mscaler.transform(X[:,1])
    mscaler.fit(y)
    y_scaled = mscaler.transform(y)
    
#    print(X,X.shape)
#    print(y,y.shape)
    arr = model.train(X,y_scaled,0.1,0.35,50,"batch")
    print("weights: ",model.W)
    print("Total Cost: ",model.cost)
    
    
    W_arr = np.array(model.W_arr)
    res = 25
    
    xx = np.linspace(np.min(W_arr[:,1])-0.2, np.max(W_arr[:,1])+0.2, res)
    yy = np.linspace(np.min(W_arr[:,2])-0.2, np.max(W_arr[:,2])+0.2, res)

    r,s = np.meshgrid(xx,yy)
    W_mesh = np.concatenate((np.ones(r.ravel().shape).reshape(-1,1)*1, r.ravel().reshape(-1,1),s.ravel().reshape(-1,1)), axis=1)
    z = np.ndarray((W_mesh.shape[0],))
    for i in range(z.shape[0]):
        z[i] = model.get_cost(model.add_bias(X),y_scaled,W_mesh[i].reshape(-1,1)) 
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(r, s, z.reshape(res,res),cmap='binary')
    ax.scatter(W_arr[:,1], W_arr[:,2], model.cost_arr)
    plt.show()
    
    plt.contour(r,s,z.reshape(res,res),levels=50)
    plt.scatter(W_arr[:,1].ravel(),W_arr[:,2].ravel(),c=model.cost_arr)
    plt.show()
