import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from preprocessing import NormalScaler
    
def likelihood(x, n, cov, mu, cov_det, cov_inv):
    """
        Likelihood of x given y = k
    """
    p = 1/(((2*np.pi)**(n/2))*((cov_det)**(0.5)))
    p *= np.exp(-0.5*(np.matmul(np.matmul((x-mu).T,cov_inv), x-mu)))
    return p[0][0]
    

if __name__ == "__main__":
    # data input
    data = pd.read_excel("./data3.xlsx",header=None)
    
    data = data.sample(frac=1).reset_index(drop=True)
    # print(data)
    X = data[[0,1,2,3]]
    y = data[4]-1
    
    # data preprocessing
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X[j])
        X[j] = mscaler.transform(X[j])
    
    # splitting data using holdout cross validation
    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0])].values
    y_train = y[:int(train_percent*X.shape[0])].values
    X_test = X[int(train_percent*X.shape[0]):].values
    y_test = y[int(train_percent*X.shape[0]):].values
    
    X_train_y1 = X_train[y_train==0]
    N_y1 = X_train_y1.shape[0]
    cov_y1 = np.cov(X_train_y1.T)
    cov_det_y1 = np.linalg.det(cov_y1)
    cov_inv_y1 = np.linalg.inv(cov_y1)
    mu_y1 = np.mean(X_train_y1 ,axis=0).reshape(-1,1)
    
    
    X_train_y2 = X_train[y_train==1]
    N_y2 = X_train_y2.shape[0]
    cov_y2 = np.cov(X_train_y2.T)
    cov_det_y2 = np.linalg.det(cov_y2)
    cov_inv_y2 = np.linalg.inv(cov_y2)
    mu_y2 = np.mean(X_train_y2,axis=0).reshape(-1,1)
    
    y_test_pred = np.ndarray((y_test.shape))

    p_y1 = X_train_y1.shape[0]/X_train.shape[0]
    p_y2 = X_train_y2.shape[0]/X_train.shape[0]
    
    for i in range(X_test.shape[0]):
        px_y1 = likelihood(X_test[i].reshape(-1,1), N_y1, cov_y1, mu_y1, cov_det_y1, cov_inv_y1)
        px_y2 = likelihood(X_test[i].reshape(-1,1), N_y2, cov_y2, mu_y2, cov_det_y2, cov_inv_y2)
        y_test_pred[i] = ((px_y1/px_y2) < (p_y2/p_y1))
        
    print("Accuracy: ", sum(y_test==y_test_pred)/y_test.shape[0])
    print("Sensitivity: ", sum((y_test==1) & (y_test_pred==1))/sum(y_test==1))
    print("Specificity: ", sum((y_test==0) & (y_test_pred==0))/sum(y_test==0))
        
    
    