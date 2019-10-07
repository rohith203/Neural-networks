import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import NormalScaler
from mpl_toolkits.mplot3d import Axes3D

def likelihood(x, n, cov, mu, cov_det, cov_inv):
    """
        Likelihood of x given y = k
    """
    p = 1/(np.power(2*np.pi,(n/2))*np.power(cov_det,0.5))
    p *= np.exp(-0.5*np.matmul(np.matmul((x-mu).T,cov_inv), x-mu))
    return p[0][0]

if __name__ == "__main__":
    data = pd.read_excel("./data4.xlsx",header=None)
    data = data.sample(frac=1).reset_index(drop=True)

    X = data[[i for i in range(7)]]
    y = data[7]

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    # data preprocessing
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X[j])
        X[j] = mscaler.transform(X[j])

    # splitting data using holdout cross validation
    train_percent = 0.7
    X_train = X[:int(train_percent*X.shape[0])].values
    y_train = y[:int(train_percent*X.shape[0])].values
    
    X_test = X[int(train_percent*X.shape[0]):].values
    y_test = y[int(train_percent*X.shape[0]):].values
    
    y_test_pred = np.ndarray((y_test.shape[0], num_classes))
    y_test_t = np.ndarray((y_test.shape[0]))

    for i in range(y_test.shape[0]):
        x = X_test[i].T
        for j in range(num_classes):
            tmp = X_train[y_train==unique_classes[j]]
            n = tmp.shape[0]
            cov = np.cov(tmp.T)
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
            mu = np.mean(tmp ,axis=0).reshape(-1,1)
            y_test_pred[i][j] = likelihood(x, n, cov, mu, cov_det, cov_inv)
    
        y_test_t[i] = unique_classes[np.argmax(y_test_pred[i])]
    
    # printing confusion matrix
    conf_mat = np.ndarray((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            conf_mat[i][j] = sum((y_test_t==unique_classes[i]) & (y_test==unique_classes[j]))
    
    print(conf_mat)