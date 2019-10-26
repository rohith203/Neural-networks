from KMeans import KMeans
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from preprocessing import NormalScaler

def gaussian(x, mu, std):
    return np.sum(np.exp((-1/(2*std**2)) * np.sum((x - mu)**2 ))    )

def multiquadric(x, mu, std):
    return np.sum(np.sqrt(np.sum(np.square(x-mu)) + np.square(std)))

def linear(x, mu, std):
    return np.sum(abs(x - mu))

def train(X_train, y_train):
    # m = number of feature vectors
    m = X_train.shape[0]
    # n = number of features
    n = X.shape[1]
    kmeans = KMeans(n_clusters=k, max_iter=max_iter , random_state=0).fit(X_train)
    
    means = kmeans.cluster_centers_
    assignments = kmeans.predict(X_train)
    stds = []
    for i in range(k):
        stds.append(np.std(X_train[(assignments==i)]))
    stds = np.array(stds)

    H = np.ndarray((m,k))
    for i in range(m):
        for j in range(k):
            H[i][j] = kernel(X_train.loc[i], means[j], stds[j])
    W = np.dot(np.linalg.pinv(H),y_train)
    return {'W': W, 'means':means, 'stds':stds}

def test(X_test, y_test, W, means, stds):
    # testing data
    mt = X_test.shape[0]
    Ht = np.ndarray((mt,k))
    for i in range(mt):
        for j in range(k):
            Ht[i][j] = kernel(X_test.loc[i], means[j], stds[j])
    yt = np.dot(Ht,W)

    cost = np.mean((yt-y_test)**2)
    print(f'cost: {cost}')
    acc = np.sum(((yt>=0)==y_test))/X_test.shape[0]
    print(f'accuracy = {acc}')
    with open('./Results/rbfnn/log.txt','a+') as log_file:
        log_file.write(f'\n\n k = {k} function = {kernel} max_iter = {max_iter} cost = {cost}  accuracy = {acc}')

if __name__=='__main__':
    # data input
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    data = data.sample(frac=1).reset_index(drop=True)
    X = data.loc[:,[i for i in range(72)]]
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X.loc[:,j])
        X.loc[:,j] = scaler.transform(X.loc[:,j])

    kernel = gaussian

    # splitting data
    max_iter = 20
    k = 4
    split = 'kfold'

    if split=='holdout':
        train_percent = 0.6
        X_train = X.iloc[:int(train_percent*X.shape[0]),:]
        y_train = y_cat[:int(train_percent*X.shape[0])]
        X_test = X.iloc[int(train_percent*X.shape[0]):, :]
        X_test = X_test.reset_index(drop=True)
        y_test = y_cat[int(train_percent*X.shape[0]):]
        
        params = train(X_train, y_train)
        test(X_test, y_test, params['W'], params['means'], params['stds'])

    elif split=='kfold':
        k_fold = 4
        Nk = X.shape[0]//k_fold
        for i in range(0, X.shape[0], Nk):
            X_test = X.loc[i:i+Nk,:].reset_index(drop=True)
            X_train = np.delete(X,range(i,i+Nk),0)
            y_test = y_cat[i:i+Nk]
            y_train = np.delete(y_cat,range(i,i+Nk),0)

            params = train(X_train, y_train)
            test(X_test, y_test, params['W'], params['means'], params['stds'])