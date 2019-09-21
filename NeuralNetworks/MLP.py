import numpy as numpy
import pandas as pd
from scipy.io import loadmat
import numpy as np

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
num_layers = 3
alpha = 0.01

# number of neurons in each layer
L = [n, 5,4,2]

# list of weight matrices of each layer excluding bias weights
W_list = []
bias_weights = []
for l in range(num_layers):
    bias_weights.append(np.random.randn())
    W_list.append(np.random.randn( L[l+1], L[l] ))

# list of output vectors of each layer excluding bias (1)
A = []
for l in range(num_layers+1):
    A.append(np.ndarray(shape=( L[l], 1 )))

# activation functions
def relu(x):
    return 0 if x<0 else x

def sigmoid(x):
    return 1/(1-np.exp(-x))

# vectorized versions of activation functions
relu = np.vectorize(relu)
sigmoid = np.vectorize(sigmoid)

def cost(z,y):
    c = 0.5*(z-y)**2
    return c

error_mat = []
for l in range(num_layers+1):
    error_mat.append(np.zeros(shape=( L[l], 1 )))

# forward propogation
A[0] = X[0].reshape(-1,1)

batch_size = 1

for l in range(1,num_layers+1):
    if l<num_layers:
        A[l] = relu(np.matmul(W_list[l-1], A[l-1]) + bias_weights[l-1])
    else:
        A[l] = sigmoid(np.matmul(W_list[l-1], A[l-1]) + bias_weights[l-1])

# backpropogation
error_mat[num_layers] = y_cat[0].reshape(-1,1) - A[num_layers] 

# update output weights
gradb = error_mat[num_layers]* A[num_layers]*(1-A[num_layers])
bias_weights[num_layers] -= (2/(2*m))*sum(gradb)
gradw = (error_mat[num_layers]* A[num_layers]*(1-A[num_layers]))
# W_list[num_layers] = W_list[num_layers] - gradb


# for l in range(num_layers-1,-1,-1):
#     bias_weights[l] -= alpha*()s
#     W_list.append(np.random.randn( L[l+1], L[l] ))