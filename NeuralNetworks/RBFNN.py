from KMeans import KMeans
import pandas as pd
import numpy as np
from scipy.io import loadmat
from preprocessing import NormalScaler


def start():
        # m = number of feature vectors
        m = X_train.shape[0]
        # n = number of features
        n = X.shape[1]

        max_iter = 2
        k = 4
        kmeans = KMeans(X_train, k)
        kmeans.run(max_iter)
        assignments = kmeans.assignments

        means = []
        stds = []
        for i in range(k):
            means.append(np.mean(X_train.loc[assignments[assignments==i].index]))
            stds.append(np.std(X_train.loc[assignments[assignments==i].index]))
        means = np.array(means)
        stds = np.array(stds)

        H = np.ndarray((m,k))
        for i in range(m):
            for j in range(k):
                H[i][j] = np.sum((X_train.loc[i].values - means[j])**3)
                # H[i][j] = np.exp((-1/(2*stds[j]**2)) * np.sum((X_train.loc[i].values - means[j])**2 ))
                # H[i][j] = np.sum((X_train.loc[i].values - means[j])**3)
        print(y_train.shape)
        W = np.dot(np.linalg.pinv(H),y_train)

        # testing data
        mt = X_test.shape[0]
        Ht = np.ndarray((mt,k))
        for i in range(mt):
            for j in range(k):
                Ht[i][j] = np.sum((X_test.loc[i].values - means[j])**3)
                # Ht[i][j] = np.exp((-1/(2*stds[j]**2)) * np.sum((X_train.loc[i].values - means[j])**2) )
        yt = np.dot(Ht,W)

        cost = np.mean((yt-y_test)**2)
        print(f'cost: {cost}')
        acc = np.sum(((yt>=0)==y_test))/X_test.shape[0]
        print(acc)
        with open('./Results/rbfnn/log.txt','a+') as log_file:
            log_file.write(f'\n\n k = {k} function = {function} max_iter = {max_iter} cost = {cost}  accuracy = {acc}')

if __name__=='__main__':
    # data input
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    X = data[[i for i in range(72)]]
    y = data.loc[:,72:73].values
    # y_cat = y.copy()
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X.loc[:,j])
        X.loc[:,j] = scaler.transform(X.loc[:,j])

    function = 'cubic'

    # splitting data
    split = 'holdout'
    if split=='holdout':
        train_percent = 0.6
        X_train = X[:int(train_percent*X.shape[0])]
        y_train = y_cat[:int(train_percent*X.shape[0])]
        X_test = X[int(train_percent*X.shape[0]):]
        X_test = X_test.reset_index(drop=True)
        y_test = y_cat[int(train_percent*X.shape[0]):]
        
        start()


    elif split=='kfold':
        k_fold = 4
        Nk = X.shape[0]//k_fold
        models = []
        for i in range(0, X.shape[0], Nk):
            X_test = X.loc[i:i+Nk,:].reset_index(drop=True)
            X_train = np.delete(X,range(i,i+Nk),0)
            y_test = y_cat[i:i+Nk]
            y_train = np.delete(y_cat,range(i,i+Nk),0)

            models.append(start())