import numpy as np
import pandas as pd
from scipy.io import loadmat

# data input
data = pd.DataFrame(loadmat('./data5.mat')['x'])
X = data.loc[:,:71].values
y = data.loc[:,72:73].values
y_cat = np.zeros((y.shape[0],2))
for i in range(y.shape[0]):
    y_cat[i][int(y[i])] = 1

# m = number of feature vectors
# n = number of features
m = X.shape[0]
n = X.shape[1]

# number of layers excluding output layer
L = 2
alpha = 0.1

# number of neurons in each layer
Layers = [n,1, 2]

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
    return 1/(1-np.exp(-x))

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
    c = 0
    cost = 0
    for i in range(m):
        # forward propogation
        forward_prop(i)
        cost += np.mean((A[L]-y_cat[i].reshape(-1,1))**2)
        c += sum(((A[L][0]<=A[L][1])==y[i].reshape(-1,1)))
    print(c,cost)


batch_size = m
for iteration in range(6):
    for i in range(0,m-batch_size+1,batch_size):
        for l in range(L):grad_b[l]=0
        for x in range(L): grad_W[x].fill(0)
        for l in range(L+1):dJA[l].fill(0)

        for j in range(i,i+batch_size):
            # forward propogation
            forward_prop(j)

            # print(A[L])

            # Backpropogation of errors
            dJA[L] = A[L] - y_cat[j].reshape(-1,1)
            for l in range(L-1, 0,-1):
                dJA[l] = np.matmul(W[l].T, A[l+1]*(1-A[l+1])*dJA[l+1])
        
            for l in range(L, 0,-1):
                grad_b[l-1] = A[l]*(1-A[l]) * dJA[l]
                grad_W[l-1] = np.matmul(A[l]*(1-A[l])*dJA[l], A[l-1].T)
        # print(dJA[1:])
        # print(grad_W[1:])
        for l in range(L, 0,-1):
            bias_weights[l-1] -= alpha*(grad_b[l-1])
            W[l-1] -= (alpha)*grad_W[l-1]

        # print(W[1])
    print("iteration: {0}".format(iteration))
    
    cnt()