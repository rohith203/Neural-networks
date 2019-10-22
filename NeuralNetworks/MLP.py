import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import NormalScaler

class MLP:

    def __init__(self, layers_arr, activ_arr):
        '''
        This is the constructor for the class MLP.
        The following attributes are initialized in this constructor
        W_list: a list of weight matrices. Each weight matrix 
                connects one layer to the next one.
        b_list: a list of bias weights. Each layer has a bias weight.
        grad_W: list of gradient matrices. Each matrix has gradients 
                of the J wrt weights in the corresponding weight matrix in W_list.
        b_grad: list of gradients of J wrt biases.
        A: a list of vectors. Each vector represents a layer of neurons and their values.
        derr: a list of vectors containing errors generated using backpropogation.
        L: number of layers excluding the input layer
        activ_arr: contains activation function to be used for a particular layer

        '''
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
        '''
        This function computes the output of layer of neurons before activation.
        '''
        return np.matmul(self.W_list[l-1], self.A[l-1]) + self.b_list[l-1]

    def activation(self, Z, activ, deriv = 0):
        '''
        This function return the activated output of a layer of neurons.
        '''
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
        '''
        This function takes ith data vector and propogates
        it forward in the neural network.
        '''
        self.A[0] = X_i.reshape(-1,1)
        for l in range(1,self.L+1):
            self.A[l] = self.activation(self.compute_Z(l), self.activ_arr[l-1])
    
    def train(self, X, y, alpha, batch_size, max_iter):
        '''
        This function takes the training data and target values,
        applies forward propogation, then applies backward propogation
        to update the weight matrices.
        mini-batch gradient descent has been used to update weights.
        '''
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
                    self.b_list[l-1] -= (alpha/batch_size)*self.grad_b[l-1]
                    self.W_list[l-1] -= (alpha/batch_size)*self.grad_W[l-1]
            
            print("iteration: {0} ".format(iteration),end="  ")
            print(" ",self.eval_cost(X,y),end=" ")
            print("  ",self.accuracy(X,y)," ")

    def eval_cost(self, X, y):
        cost = 0

        for i in range(y.shape[0]):
            # forward propogation
            self.forward_prop(X[i])
            cost += np.sum((self.A[self.L]-y[i].reshape(-1,1))**2)
        return cost

    def accuracy(self, X, y):
        acc = 0
        for i in range(y.shape[0]):
            # forward propogation
            self.forward_prop(X[i])
            t1 = 0 if self.A[self.L][0][0]<0.5 else 1
            t2 = 0 if self.A[self.L][1][0]<0.5 else 1

            acc += ((t1==y[i][0]) and (t2==y[i][1]))
        return acc/y.shape[0]

    def conf_mat(self, X, y):
        conf_mat = np.zeros((y.shape[1],y.shape[1]))
        y_p = self.predict(X)
        for i in range(y.shape[0]):
            # forward propogation
            conf_mat[int(np.argmax(y[i]))][int(y_p[i])] += 1
        return conf_mat
    
    def predict(self, X):
        y_pred = np.ndarray(X.shape[0])
        for i in range(X.shape[0]):
            # forward propogation
            self.forward_prop(X[i])
            y_pred[i] = np.argmax(self.A[self.L])
        return y_pred

    
if __name__=='__main__':
        
    # data input
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.loc[:,:71].values
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2)).astype(np.int)
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X[:,j])
        X[:,j] = scaler.transform(X[:,j])

    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0])]
    y_train = y_cat[:int(train_percent*X.shape[0])]
    X_test = X[int(train_percent*X.shape[0]):]
    y_test = y_cat[int(train_percent*X.shape[0]):]

    # m = number of feature vectors
    m = X_train.shape[0]
    # n = number of features
    n = X_train.shape[1]

    alpha = 0.5
    max_iter = 25

    # number of neurons in each layer
    Layers = [n,16,8,2]
    activations = ['sigmoid','sigmoid','sigmoid']
    # Layers = [n,12,1]
    batch_size = 32

    model = MLP(Layers, activations)
    model.train(X_train, y_train, alpha, batch_size, max_iter)

    # print("Test Accuracy", model.accuracy(X_test,y_test))
    conf = model.conf_mat(X_test,y_test)
    print("Confusion matrix", conf)
    print("accuracy_conf: ", (conf[0][0]+conf[1][1])/(np.sum(conf)))
    print("accuracy: ", model.accuracy(X_test,y_test))