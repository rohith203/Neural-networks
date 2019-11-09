import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import NormalScaler

class ELM:
    def __init__(self, L, X, Y, activation="tanh"):
        '''
        In this constructor the hidden layer matrix
        H is made using random weights a and b.
        m: number of samples
        n: number of features
        L: number of neurons in the hidden layer
        '''
        m = X.shape[0]
        n = X.shape[1]
        self.L = L
        H = np.ndarray((m, L), dtype=np.float64)
        self.a = np.random.rand(L, n)
        self.b = np.random.rand(L,1)
        if(activation=="sigmoid"): self.activate = self.sigmoid
        elif(activation=="gaussian"): self.activate = self.gaussian
        elif(activation=="tanh"): self.activate = self.tanh

        for i in range(m):
            for j in range(L):
                H[i][j] = self.activate(X[i], self.a[j], self.b[j])

        self.train(H,Y)

    def train(self, H, Y):
        '''
        This function uses the following vectorized formula
        to compute and return the weight matrix between 
        hidden and output layer.
        '''
        self.W = np.dot(np.linalg.pinv(H), Y)
        return self.W

    def test(self, X_test, y_test):
        '''
        This function computes the predicted values with
        the given test feature vectors.
        '''
        y_pred = y_test.copy()
        acc = 0
        conf_mat = np.zeros((y_test.shape[1], y_test.shape[1]))
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

            conf_mat[np.argmax(y_test[i])][np.argmax(y_pred[i])] +=1

        acc = acc/y_pred.shape[0]
        print(acc)
        print(conf_mat)
        return (acc, y_pred)
    


    def gaussian(self,x,a,b):
        t = -b*np.linalg.norm(x.reshape(-1,1)-a.reshape(-1,1), 1)
        r = np.exp(t)
        return r
    
    def tanh(self,x,a,b):
        tmp = np.exp(-(np.dot(x.reshape(1,-1), a.reshape(1,-1).T)[0][0] + b))
        return (1-tmp)/(1+tmp)
    
    def sigmoid(self, x,a,b):
        return 1/(1+np.exp(-(np.dot(x.reshape(1,-1), a.reshape(1,-1).T)[0][0] + b)))


if __name__=='__main__':
            
    # data input
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    data = data.sample(frac=1)
    X = data.loc[:,:71].values
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X[:,j])
        X[:,j] = scaler.transform(X[:,j])

    k_fold = 5
    Nk = X.shape[0]//k_fold
    models = []
    acc = []
    iterat = 1
    for i in range(0, X.shape[0]-Nk+1, Nk):
        print("\n\nk fold iteration: ", iterat)
        X_test = X[i:i+Nk,:]
        X_train = np.delete(X,range(i,i+Nk),0)
        y_test = y_cat[i:i+Nk,:]
        y_train = np.delete(y_cat,range(i,i+Nk),0)

        # m = number of feature vectors
        m = X_train.shape[0]
        # n = number of features
        n = X_train.shape[1]
        L = 128
        L = 300
        elm = ELM(L, X_train, y_train, "gaussian")
        models.append(elm)
        acc.append(elm.test(X_test,y_test)[0])
        print(f"Train accuracy,confmat: {elm.test(X_train, y_train)[0]}")
        iterat+=1

    print("Average Accuracy: ", np.mean(acc))
    