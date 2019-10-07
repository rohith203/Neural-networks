import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import NormalScaler

# data input
data = pd.DataFrame(loadmat('./data5.mat')['x'])
X = data.loc[:,:71].values
y = data.loc[:,72:73].values
y_cat = y.copy()
# y_cat = np.zeros((y.shape[0],2))
# for i in range(y.shape[0]):
#     y_cat[i][int(y[i])] = 1

# data preprocessing
scaler = NormalScaler()
for j in range(X.shape[1]):
    scaler.fit(X[:,j])
    X[:,j] = scaler.transform(X[:,j])

# m = number of feature vectors
# n = number of features
m = X.shape[0]
n = X.shape[1]

# number of layers excluding output layer
L = 3
alpha = 0.04

# number of neurons in each layer
Layers = [n,6,3,1]

# list of weight matrices of each layer excluding bias weights
W = []
grad_W = []
bias_weights = []
grad_b = []
for l in range(L):
    bias_weights.append(np.random.randn())
    grad_b.append(np.random.randn())
    W.append(np.random.randn( Layers[l+1], Layers[l] ))
    grad_W.append(np.random.randn( Layers[l+1], Layers[l] ))

# list of output vectors of each layer excluding bias (1)
A = []
dJA = []
for l in range(L+1):
    A.append(np.ndarray(shape=( Layers[l], 1 )))
    dJA.append(np.ndarray(shape=( Layers[l], 1 )))

# activation functions
def relu(x):
    return 0 if x<0 else x

def sigmoid(x):
    return 1/(1+np.exp(-x))

# vectorized versions of activation functions
relu = np.vectorize(relu)
sigmoid = np.vectorize(sigmoid)

def forward_prop(i):
    A[0] = X[i].reshape(-1,1)
    for l in range(1,L+1):
        if l<L:
            A[l] = sigmoid(np.matmul(W[l-1], A[l-1]) + bias_weights[l-1])
        else:
            A[l] = sigmoid(np.matmul(W[l-1], A[l-1]) + bias_weights[l-1])

def cnt():
    # c = 0
    cost = 0
    print("iteration: {0}".format(iteration))

    for i in range(m):
        # forward propogation
        forward_prop(i)
        # print(A[L])
        t = 0 if A[L][0][0]<0.5 else 1
        cost += np.sum((t-y_cat[i].reshape(-1,1))**2)
        # c += sum(((A[L][0]<=A[L][1])==y[i].reshape(-1,1)))
    print(cost)

bsum = bias_weights.copy()
wsum = W.copy()
batch_size = 1
for iteration in range(30):
    for i in range(0,m-batch_size+1,batch_size):
        for l in range(L): grad_b[l]=0
        for x in range(L): grad_W[x].fill(0)
        # for l in range(L+1):dJA[l].fill(0)

        for j in range(i,i+batch_size):
            # forward propogation
            forward_prop(j)

            # Backpropogation of errors
            for k in range(A[L].shape[0]):
                dJA[L][k] = (A[L][k]-y_cat[j].reshape(-1,1)[k]) * A[L][k] * (1-A[L][k])
            for l in range(L-1, 0,-1):
                s = 0
                for p in range(A[l].shape[0]):
                    for q in range(A[l+1].shape[0]):
                        
                for k in range(A[l].shape[0]):
                    dJA[l][k] = A[l][k]*(1-A[l][k])*W[l].T, dJA[l+1])
                
        
            for l in range(L, 0,-1):
                grad_b[l-1] += dJA[l]
                grad_W[l-1] += np.matmul(dJA[l], A[l-1].T)

            # for l in range(L, 0,-1):
            #     bsum[l-1] += grad_b[l-1]
            #     wsum[l-1] += grad_W[l-1]
        
        for l in range(L, 0,-1):
            bias_weights[l-1] -= (alpha/batch_size)*grad_b[l-1]
            W[l-1] -= (alpha/batch_size)*grad_W[l-1]

    cnt()
    # print(W[2])