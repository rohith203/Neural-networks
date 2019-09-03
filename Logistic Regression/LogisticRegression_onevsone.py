import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class LogisticRegression:
    
    def init_weights(self, s):
        np.random.seed(11)
        self.W = np.random.randn(s[1],1)
        self.W_arr = []
        self.cost_arr = []
        self.cost = 0
        self.gradient = 0
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
        
    def get_cost(self, X, y, W):
        total_cost = 0
        for i in range(X.shape[0]):
            total_cost += y[i]*np.log(self.get_h_i(X, i, W)) + (1-y[i])*np.log(1-self.get_h_i(X, i, W))
        return (0.5/X.shape[0])*total_cost
    
    def add_bias(self, X):
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X), axis=1)
    
    def get_h_i(self, X, i, W):
        h_i = np.matmul(X[i].reshape(1,-1),W)
        return self.sigmoid(h_i[0][0])

    def batch_grad_descent(self, X, y, alpha, max_iter):
        for iteration in range(max_iter):
            W_new = np.ndarray(self.W.shape)
            for j in range(X.shape[1]):
                grad = 0
                for i in range(X.shape[0]):
                    grad += (self.get_h_i(X, i, self.W) - y[i])*X[i][j]
                W_new[j][0] = self.W[j][0] - (alpha/X.shape[0])*grad
            self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
        return W_new
    
    def stochastic_grad_descent(self, X, y, alpha, max_iter):
        for iteration in range(max_iter):
            W_new = self.W.copy()
            for i in range(X.shape[0]):    
                ind = np.random.randint(X.shape[0])
                grad = 0
                for j in range(X.shape[1]):
                    grad += (self.get_h_i(X, ind, self.W) - y[ind]) * X[ind][j]
                for j in range(X.shape[1]):
                    W_new[j][0] = self.W[j][0] - (alpha)*grad
                self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
        return self.W
                              
        
    def train(self, X, y, alpha, max_iter=100, option="batch"):
        X = self.add_bias(X)
        self.init_weights(X.shape)
        if option=="batch":
            self.batch_grad_descent(X,y,alpha,max_iter)
        elif option=="stochastic":
            self.stochastic_grad_descent(X,y,alpha,max_iter)
        self.cost = self.cost_arr[len(self.cost_arr)-1]
        return self.cost_arr
        
    def test(self,X,W=""):
        if W=="":W = self.W

        X = self.add_bias(X)
        y_pred = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.get_h_i(X,i,W)
            for j in range(X.shape[1]):
                y_pred[i] += X[i][j]*W[j][0]
        return self.sigmoid(y_pred)

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
    model = LogisticRegression()

    data = pd.read_excel("./data4.xlsx",header=None)
    # print(data)
    X = data[[i for i in range(7)]]
    y = data[7]
    
    # data preprocessing
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X[j])
        X[j] = mscaler.transform(X[j])
    
#    model.train(X,y,0.1,100,'batch')   
#
#    y_pred = model.test(X)
#    for i in range(y_pred.shape[0]):
#        y_pred[i] = 0 if y_pred[i]<0.5 else 1
#
#    print(y_pred)
#
#    from sklearn.metrics import accuracy_score
#    print(accuracy_score(y,y_pred))

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            class_i = unique_classes[i]
            class_j = unique_classes[j]
            
            model = LogisticRegression()
            y_i_j = y.apply(lambda x:1 if x==class_i else 0)
            model.train(X,y_i,0.1,100,'batch')
            y_pred = model.test(X)
            for i in range(y_pred.shape[0]):
                y_pred[i] = 0 if y_pred[i]<0.5 else 1
            print("Class: ", class_i, " vs all")
            print(y_pred)
        
            from sklearn.metrics import accuracy_score
            print(accuracy_score(y_i,y_pred))
            
    for class_i in unique_classes:
        for class_j in unique_classes:
        model = LogisticRegression()
        y_i = y.apply(lambda x:1 if x==class_i else 0)
        model.train(X,y_i,0.1,100,'batch')
        y_pred = model.test(X)
        for i in range(y_pred.shape[0]):
            y_pred[i] = 0 if y_pred[i]<0.5 else 1
        print("Class: ", class_i, " vs all")
        print(y_pred)
    
        from sklearn.metrics import accuracy_score
        print(accuracy_score(y_i,y_pred))