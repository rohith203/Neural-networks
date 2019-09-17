import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression_Vectorized:
    
     def init_weights(self, s):
         np.random.seed(11)
         self.W = np.random.randn(s[1],1)
         self.W_arr = []
         self.cost_arr = []
         self.cost = 0
         self.gradient = 0
        
     def get_cost(self, X, y, W):
         return (0.5/X.shape[0])*np.sum(np.square(np.matmul(X,W)-y.reshape(-1,1)))
    
     def add_bias(self, X):
         bias = np.ones((X.shape[0],1))
         return np.concatenate((bias,X), axis=1)
                                  
     def train(self, X, y):
         X = self.add_bias(X)
         self.init_weights(X.shape)
         self.W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), y.reshape(-1,1))
         self.cost = self.get_cost(X,y,self.W)
         return self.W
        
     def test(self,X):
         X = self.add_bias(X)
         return np.matmul(X,self.W)
        

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
    model = LinearRegression_Vectorized()
    
    data = pd.read_csv("./data.csv", header=None)
    X = data.loc[:,0:1].values
    y = data.loc[:,2].values
    mscaler = NormalScaler()
    mscaler.fit(X[:,0])
    X[:,0] = mscaler.transform(X[:,0])
    mscaler.fit(X[:,1])
    X[:,1] = mscaler.transform(X[:,1])
#    mscaler.fit(y)
#    y_scaled = mscaler.transform(y)

    arr = model.train(X,y)
    print("weights: ",model.W)
    print("Total Cost: ",model.cost)

    xx = np.linspace(np.min(X[:,0]),np.max(X[:,0]),25)
    yy = np.linspace(np.min(X[:,1]),np.max(X[:,1]),25)
    r,s = np.meshgrid(xx,yy)
    zz = np.concatenate((r.ravel().reshape(-1,1),s.ravel().reshape(-1,1)),axis=1)
    z = model.test(zz)
    ax = plt.axes(projection='3d')
    ax.scatter(X[:,0],X[:,1],y)
    ax.plot_surface(r,s,z.reshape(xx.shape[0],xx.shape[0]),cmap='coolwarm')
    plt.show()