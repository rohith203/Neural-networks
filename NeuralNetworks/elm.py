import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import NormalScaler

class ELM:
    def __init__(self, L, X, Y, activation="sigmoid"):
        m = X.shape[0]
        n = X.shape[1]
        self.L = L
        H = np.ndarray((m, L))
        self.a = np.random.randn(L, n)
        self.b = np.random.randn(L)
        if(activation=="sigmoid"): self.activate = self.sigmoid
        elif(activation=="gaussian"): self.activate = self.gaussian
        elif(activation=="tanh"): self.activate = self.tanh

        for i in range(m):
            for j in range(L):
                H[i][j] = self.activate(X[i], self.a[j], self.b[j])

        self.train(H,y_cat)

    def train(self, H, Y):
        self.W = np.dot(np.linalg.pinv(H), y_cat)
        return self.W

    def test(self, X_test, y_test):
        y_pred = y_test.copy()
        acc = 0
        for i in range(X_test.shape[0]):
            x = []
            for j in range(self.L):
                x.append(self.activate(X_test[i], self.a[j], self.b[j]))
            x = np.array(x)
            max_i = np.argmax(np.dot(x.reshape(1,-1), self.W))

            for j in range(y_pred.shape[1]):
                if j==max_i: 
                    y_pred[i][j] = 1
                    if max_i==np.argmax(y_test[i]):
                        acc+=1
                else: y_pred[i][j] = 0

        print("accuracy: ", acc/y_pred.shape[0])
        return y_pred

    def gaussian(self,x,a,b):
        return np.exp(-b*np.sum(abs(x-a)))
    
    def tanh(self,x,a,b):
        tmp = np.exp(-(np.dot(x.reshape(1,-1), a.reshape(1,-1).T)[0][0] + b))
        return (1-tmp)/(1+tmp)
    
    def sigmoid(self, x,a,b):
        return 1/(1+np.exp(-(np.dot(x.reshape(1,-1), a.reshape(1,-1).T)[0][0] + b)))


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
    L = 12
    elm = ELM(L, X, y_cat, "sigmoid")
    elm.test(X,y_cat)