import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from preprocessing import NormalScaler

def gaussian(x, mu, std):
    if std==0: std=1e-10
    return np.exp((-1/(2*(std**2))) * np.sum(np.square(x - mu)))

def multiquadric(x, mu, std):

    return np.sum(np.sqrt(np.sum(np.square(x-mu)) + (std**2)))

def linear(x, mu, std):
    return np.sum(abs(x - mu))

def train(X_train, y_train):
    # m = number of feature vectors
    m = X_train.shape[0]
    # n = number of features
    n = X.shape[1]
    
    # running k means clustering
    kmeans = KMeans(n_clusters=k, max_iter=max_iter , random_state=11)
    kmeans.fit(X_train)
    means = kmeans.cluster_centers_
    assignments = kmeans.labels_

    stds = []
    for i in range(k):
        temp = X_train[(assignments==i)]
        t = (1/temp.shape[0])*np.sum(abs(temp-means[i]))
        stds.append(t)
    stds = np.array(stds)

    # Hidden layer matrix
    H = np.ndarray((m,k))
    for i in range(m):
        for j in range(k):
            H[i][j] = kernel(X_train[i], means[j], stds[j])

    # weight matrix
    W = np.dot(np.linalg.pinv(H),y_train)
    return {'W': W, 'means':means, 'stds':stds}

def test(X_test, y_test, W, means, stds):
    # testing data
    mt = X_test.shape[0]
    Ht = np.ndarray((mt,k))
    for i in range(mt):
        for j in range(k):
            Ht[i][j] = kernel(X_test[i], means[j], stds[j])
    yt = np.dot(Ht,W)

    # computing confusion matrix
    conf_mat = np.zeros((y_test.shape[1],y_test.shape[1]))
    for i in range(y_test.shape[0]):
        a = int(np.argmax(y_test[i]))
        b = int(np.argmax(yt[i]))
        conf_mat[a][b] += 1
        
    cost = np.mean((yt-y_test)**2)
    print(f'cost: {cost}')
    acc = 0
    for i in range(y_test.shape[0]):
        acc += (np.argmax(yt[i]) == np.argmax(y_test[i]))
    acc /=y_test.shape[0]
    print(f'accuracy = {acc}')
    print(f'confusion matrix\n {conf_mat}')
    with open('./Results/rbfnn/log.txt','a+') as log_file:
        log_file.write(f'\n\niteration: {iteration}\n k = {k}\nfunction = {kernel}\nmax_iter = {max_iter}\n cost = {cost}  accuracy = {acc} \nconf_mat=\n{conf_mat}')
    return (cost, acc)

if __name__=='__main__':
    # data input
    data = pd.DataFrame(loadmat('./data5.mat')['x'])
    data = data.sample(frac=1).reset_index(drop=True)

    X = data.loc[:,[i for i in range(72)]].values
    y = data.loc[:,72:73].values
    y_cat = np.zeros((y.shape[0],2))
    for i in range(y.shape[0]):
        y_cat[i][int(y[i])] = 1

    # data preprocessing
    scaler = NormalScaler()
    for j in range(X.shape[1]):
        scaler.fit(X[:,j])
        X[:,j] = scaler.transform(X[:,j])

    kernel = gaussian
    # kernel = multiquadric
    # kernel = linear

    # splitting data
    max_iter = 30
    k = 50
    split = 'holdout'

    if split=='holdout':
        iteration = 1
        train_percent = 0.6
        X_train = X[:int(train_percent*X.shape[0]),:]
        y_train = y_cat[:int(train_percent*X.shape[0])]
        X_test = X[int(train_percent*X.shape[0]):, :]
        y_test = y_cat[int(train_percent*X.shape[0]):]
        
        params = train(X_train, y_train)
        test(X_test, y_test, params['W'], params['means'], params['stds'])

    elif split=='kfold':
        k_fold = 5
        Nk = X.shape[0]//k_fold
        accs = []
        iteration = 1
        for i in range(0, X.shape[0]-Nk+1, Nk):
            print(f'K-fold {i} {i+Nk}: ', iteration)
            X_test = X[i:i+Nk,:]
            X_train = np.delete(X,range(i,i+Nk),0)
            y_test = y_cat[i:i+Nk]
            y_train = np.delete(y_cat,range(i,i+Nk),0)

            params = train(X_train, y_train)
            cost, acc = test(X_test, y_test, params['W'], params['means'], params['stds'])
            accs.append(acc)
            iteration+=1
        print("\nAvg Accuracy: ", np.mean(accs),'\n')
        with open('./Results/rbfnn/log.txt','a+') as log_file:
            log_file.write(f'\nAvg Accuracy: {np.mean(accs)}')