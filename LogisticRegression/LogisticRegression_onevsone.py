import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from LogisticRegression import LogisticRegression,NormalScaler


if __name__ == "__main__":
    model = LogisticRegression()

    data = pd.read_excel("./data4.xlsx",header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    
    X = data[[i for i in range(7)]]
    y = data[7]
    
    # data preprocessing
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X[j])
        X[j] = mscaler.transform(X[j])
        
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    num_models = (int)(num_classes*(num_classes-1)/2)
    
    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0])]
    y_train = y[:int(train_percent*X.shape[0])]
#    y_cat_train = y_cat[:int(train_percent*X.shape[0])]
    X_test = X[int(train_percent*X.shape[0]):]
    y_test = y[int(train_percent*X.shape[0]):]
#    y_cat_test = y_cat[int(train_percent*X.shape[0]):]
    
    models = [[0 for j in range(num_classes)] for i in range(num_classes)]
    
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            class_i = unique_classes[i]
            class_j = unique_classes[j]
            
            models[i][j] = LogisticRegression()
            tmp = (y_train==class_i) | (y_train==class_j)
            y_train_i_j = (y_train[tmp]==class_i).astype('int').values
            models[i][j].train(X_train[tmp], y_train_i_j, 0.1, 100, 'batch')
                        
    y_test_pred = np.ndarray((y_test.shape[0], num_models))
    k = 0
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            class_i = unique_classes[i]
            class_j = unique_classes[j]
            
            y_test_pred[:,k] = models[i][j].test(X_test)
            y_test_pred[:,k][y_test_pred[:,k]>=0.5] = class_i
            y_test_pred[:,k][y_test_pred[:,k]<0.5] = class_j
            
            k+=1
    
    y_test_t = np.ndarray((y_test.shape[0],))
    for i in range(y_test.shape[0]):
        uniqu,counts = np.unique(y_test_pred[i],return_counts=True) 
        y_test_t[i] = uniqu[np.argmax(counts)]
    print("Accuracy: ", sum(y_test_t==y_test)/y_test.shape[0])