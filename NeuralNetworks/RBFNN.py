from KMeans import KMeans
import pandas as pd
import numpy as np
from scipy.io import loadmat
from preprocessing import NormalScaler

# if __name__=='__main__':

# data input
data = pd.DataFrame(loadmat('./data5.mat')['x'])
X = data[[i for i in range(72)]]
y = data.loc[:,72:73].values
y_cat = y.copy()

# data preprocessing
scaler = NormalScaler()
for j in range(X.shape[1]):
    scaler.fit(X.loc[:,j])
    X.loc[:,j] = scaler.transform(X.loc[:,j])

train_percent = 0.7
X_train = X[:int(train_percent*X.shape[0])]
y_train = y[:int(train_percent*X.shape[0])]
X_test = X[int(train_percent*X.shape[0]):]
X_test = X_test.reset_index(drop=True)
y_test = y[int(train_percent*X.shape[0]):]

# m = number of feature vectors
m = X_train.shape[0]
# n = number of features
n = X.shape[1]

max_iter = 4
k = 5
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
W = np.dot(np.linalg.pinv(H),y_train.reshape(-1,1))

# testing data
mt = X_test.shape[0]
Ht = np.ndarray((mt,k))
for i in range(mt):
    for j in range(k):
        Ht[i][j] = np.sum((X_test.loc[i].values - means[j])**3)
yt = np.dot(Ht,W)