import numpy as np
import pandas as pd
from scipy.io import loadmat
from preprocessing import NormalScaler
from MLP_auto import MLP

if __name__=='__main__':
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
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

    # m = number of feature vectors
    m = X.shape[0]
    # n = number of features
    n = X.shape[1]

    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0])]
    y_train = y[:int(train_percent*X.shape[0])]
    X_test = X[int(train_percent*X.shape[0]):]
    y_test = y[int(train_percent*X.shape[0]):]

    # number of layers excluding output layer
    alpha = 0.5
    max_iter = 10

    # pretraining 3 autoencoders
    model11 = MLP([n, 100], ['sigmoid'])
    print("training autoencoder 1")
    model11.train(X_train,X_train, alpha, 12, max_iter)
  
    out1 = model11.output_hidden(X_train)

    model12 = MLP([100, 60], ['sigmoid'])
    print("training autoencoder 2")
    model12.train(out1, out1, alpha, 12, max_iter)
    
    out2 = model12.output_hidden(out1)

    model13 = MLP([60, 40], ['sigmoid'])
    print("training autoencoder 3")
    model13.train(out2, out2, alpha, 12, max_iter)

    # stacking the pretrained autoencoders
    model = MLP([n, 100, 60, 40], ['sigmoid','sigmoid','sigmoid'])
    
    # initializing pretrained weights
    model.W_list[0] = model11.W_list[0]
    model.W_list[5] = model11.W_list[0].T
    
    model.W_list[1] = model12.W_list[0]
    model.W_list[4] = model12.W_list[0].T
    
    model.W_list[2] = model13.W_list[0]
    model.W_list[3] = model13.W_list[0].T
    
    # finetuning the stacked autoencoder
    print("training stacked autoencoder")
    model.train(X_train, X_train, alpha, 12, 100)