import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import NormalScaler

class MLP:

    def __init__(self, layers_arr, activ_arr):
        self.W_list = []
        self.b_list = []
        self.grad_W = []
        self.grad_b = []
        self.A = []
        self.derr = []
        self.L = len(layers_arr)-1
        self.activ_arr = activ_arr
        self.n = layers_arr[0]
        for l in range(self.L):
            self.b_list.append(np.random.randn(1,).astype(np.float64)[0])
            self.grad_b.append(np.random.randn(1,).astype(np.float64)[0])
            self.W_list.append(np.random.randn( layers_arr[l+1], layers_arr[l]).astype(np.float64))
            self.grad_W.append(np.random.randn( layers_arr[l+1], layers_arr[l]).astype(np.float64))
        for l in range(self.L+1):
            self.A.append(np.ndarray(shape=( layers_arr[l], 1 )).astype(np.float64))
            self.derr.append(np.ndarray(shape=( layers_arr[l], 1 )).astype(np.float64))

        # vectorized versions of activation functions
        self.relu_v = np.vectorize(self.relu)
        self.sigmoid_v = np.vectorize(self.sigmoid)
    
    def compute_Z(self, l):
        return np.matmul(self.W_list[l-1], self.A[l-1]) + self.b_list[l-1]

    def activation(self, Z, activ, deriv = 0):
        if(activ=='sigmoid'):
            return self.sigmoid_v(Z, deriv)
        elif(activ=='relu'):
            return self.relu_v(Z, deriv)

    def relu(self, x, deriv = 0):
        if deriv==1: 
            return 0 if x<0 else 1 
        return 0 if x<0 else x

    def sigmoid(self, x, deriv = 0):
        if deriv==1:
            return self.sigmoid(x)*(1-self.sigmoid(x))
        return 1/(1+np.exp(-x))

    def forward_prop(self, X_i):
        self.A[0] = X_i.reshape(-1,1)
        for l in range(1,self.L+1):
            self.A[l] = self.activation(self.compute_Z(l), self.activ_arr[l-1])
    
    def train(self, X, y, alpha, batch_size, max_iter):
        m = y.shape[0]
        for iteration in range(max_iter):
            for i in range(0,m-batch_size+1,batch_size):
                for l in range(self.L): self.grad_b[l]=0
                for l in range(self.L): self.grad_W[l].fill(0)

                for j in range(i,i+batch_size):
                    # forward propogation
                    self.forward_prop(X[j])

                    # Backpropogation of errors
                    self.derr[self.L] = (self.A[self.L]-y[j].reshape(-1,1)) * self.activation(self.compute_Z(self.L), self.activ_arr[self.L-1], 1)
                    for l in range(self.L-1, 0,-1):
                        self.derr[l] = self.activation(self.compute_Z(l), self.activ_arr[l-1], 1)*np.matmul(self.W_list[l].T, self.derr[l+1])
                
                    for l in range(self.L, 0,-1):
                        self.grad_b[l-1] += np.mean(self.derr[l])
                        self.grad_W[l-1] += np.matmul(self.derr[l], self.A[l-1].T)
                
                for l in range(self.L, 0,-1):
                    self.b_list[l-1] -= (alpha/batch_size)*np.mean(self.grad_b[l-1])
                    self.W_list[l-1] -= (alpha/batch_size)*self.grad_W[l-1]
            
            print("iteration: {0} ".format(iteration),end="  ")
            self.eval_cost(X,y)
            print("  ",self.accuracy(X,y)," ")

    def eval_cost(self, X, y):
        cost1 = 0
        cost2 = 0

        for i in range(y.shape[0]):
            # forward propogation
            self.forward_prop(X[i])
            t1 = 0 if self.A[self.L][0][0]<0.5 else 1
            t2 = 0 if self.A[self.L][1][0]<0.5 else 1
            cost1 += (t1-y[i].reshape(-1,1)[0])**2 +(t2-y[i].reshape(-1,1)[1])**2
            cost2 += np.sum((self.A[self.L]-y[i].reshape(-1,1))**2)
        print(" ",cost1,cost2,end=" ")

    def accuracy(self, X, y):
        acc = 0
        # y = y.reshape(-1,1)
        for i in range(y.shape[0]):
            # forward propogation
            self.forward_prop(X[i])
            t1 = 0 if self.A[self.L][0][0]<0.5 else 1
            t2 = 0 if self.A[self.L][1][0]<0.5 else 1

            acc += ((t1==y[i][0]) and (t2==y[i][1]))
        return acc/y.shape[0]

    
if __name__=='__main__':
        
    # data input
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    X = data.loc[:,:71].values
    y = data.loc[:,72:73].values
    # y_cat = y.copy()
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X[:,j])
        X[:,j] = scaler.transform(X[:,j])

    # m = number of feature vectors
    m = X.shape[0]
    # n = number of features
    n = X.shape[1]

    # number of layers excluding output layer
    L = 3
    alpha = 0.5
    max_iter = 200

    # number of neurons in each layer
    Layers = [n,12,8,2]
    activations = ['sigmoid','sigmoid','sigmoid']
    # Layers = [n,12,1]
    batch_size = 12

    model = MLP(Layers, activations)
    model.train(X, y_cat, alpha, batch_size, max_iter)
