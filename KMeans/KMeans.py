import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D

class KMeans:
    def __init__(self, X, k):
        self.assignments = pd.Series([random.randint(0,k-1) for i in range(X.shape[0])])
        self.centers = X.iloc[[random.randint(0,X.shape[0]-1) for i in range(k)]]
        self.k = k
        self.cost_arr = []
        
    def get_dist(self,a,b):
        return np.sum(np.square(a-b),axis=1)
        
    def get_cost(self, X):
        cost = 0
        for i in range(self.k):
            cost += np.sum(self.get_dist(X[self.assignments==i], self.centers.iloc[i]))
        self.cost_arr.append(cost)
        return cost
    
    def run(self, max_iter=100):
        
        
        for iteration in range(max_iter):
            # assigning points to their nearest center
            for i in range(X.shape[0]):
                nearest_center_i = np.sum(np.square(X.iloc[i]-model.centers), axis=1).values.argmin()
                self.assignments[i] = nearest_center_i
            self.get_cost(X)
            # calculating mean of each cluster
            for i in range(self.k):
                self.centers.iloc[i] = np.mean(X[self.assignments==i])
            
            

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
    

#    data = pd.read_csv("./data.csv", header=None)
    data = pd.read_excel("./data2.xlsx",header=None)
    # print(data)
    X = data.copy()
#    mscaler = MinMaxScaler()
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X[j])
        X[j] = mscaler.transform(X[j])
        
    model = KMeans(X,2)
    
    model.run(50)
    
    plt.plot(model.cost_arr)
    plt.scatter(X[0],X[2],c=model.assignments)
    
    