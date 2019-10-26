import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from preprocessing import NormalScaler

class KMeans:
    # X = pd.DataFrame()
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.assignments = pd.Series([random.randint(0,k-1) for i in range(X.shape[0])])
        self.centers = X.iloc[[random.randint(0,X.shape[0]-1) for i in range(k)]]
        self.cost_arr = []
        print("init finished")
        
    def get_dist(self,a,b):
        return np.sum((a-b)**2,axis=1)
        
    def get_cost(self):
        cost = 0
        for i in range(self.k):
            cost += np.sum(self.get_dist(self.X[self.assignments==i], self.centers.iloc[i]))
        self.cost_arr.append(cost)
        return cost
    
    def run(self, max_iter=100):
        print("***KMeans Clustering started***")
        for iteration in range(max_iter):
            # assigning points to their nearest center
            print("iteration: {0}".format(iteration))
            for i in range(self.X.shape[0]):
                nearest_center_i = np.sum((self.X.iloc[i]-self.centers)**2, axis=1).values.argmin()
                self.assignments[i] = nearest_center_i
            print("centers assigned")
            # self.get_cost()
            # calculating mean of each cluster
            for i in range(self.k):
                self.centers.iloc[i] = np.mean(self.X[self.assignments==i])
            print("centers recalculated")
            if len(self.cost_arr)>1 and abs(self.cost_arr[-2]-self.cost_arr[-1])==0:
                break
            print("*****\n")
        print(self.get_cost())

# if __name__ == "__main__":
#     # data input
#     data = pd.read_excel("./data2.xlsx",header=None)
#     X = data.copy()
#     # data Normalization
#     mscaler = NormalScaler()
#     for j in range(X.shape[1]):
#         mscaler.fit(X[j])
#         X[j] = mscaler.transform(X[j])
        
#     # initializing k-means with k value
#     k = 2
#     max_iter = 50
#     model = KMeans(X,k)
#     # running k-means algorithm
#     model.run(max_iter)
#     y = model.assignments

#     print("minimum cost",model.cost_arr[-1])

#     # data visualization
#     plt.title("Cost Function vs iteration plot k={0}".format(k))
#     plt.xlabel("iteration")
#     plt.ylabel("cost")
#     plt.plot(model.cost_arr)
#     plt.savefig("./Results/k_{0}.png".format(k))
#     plt.show()

#     for i in range(4):
#         plt.title("Feature {0} vs y. k={1}".format(i+1, k))
#         plt.xlabel("Feature {0}".format(i+1))
#         plt.ylabel("y")
#         plt.scatter(X[i],y,c=model.assignments)
#         plt.savefig("./Results/k_{0}_{1}.png".format(k,i+1))
#         plt.show()