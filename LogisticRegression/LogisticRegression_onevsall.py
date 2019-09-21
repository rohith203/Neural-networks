import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from LogisticRegression import LogisticRegression,NormalScaler

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
    
    
    y_cat = (y==unique_classes[0]).astype('int').values.reshape(-1,1)
    for i in unique_classes[1:]:
        y_cat = np.concatenate((y_cat,(y==i).astype('int').values.reshape(-1,1)),axis=1)
    
    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0])]
    y_train = y[:int(train_percent*X.shape[0])]
    y_cat_train = y_cat[:int(train_percent*X.shape[0])]
    X_test = X[int(train_percent*X.shape[0]):]
    y_test = y[int(train_percent*X.shape[0]):]
    y_cat_test = y_cat[int(train_percent*X.shape[0]):]
    
    models = [LogisticRegression() for i in unique_classes]
    
    y_train_pred = np.ndarray((y_train.shape[0],num_classes))
    for c in range(num_classes):
        models[c].train(X_train,y_cat_train[:,c],0.1,100,'batch')
        y_train_pred[:,c] = models[c].test(X_train)
    
    y_test_pred = np.ndarray((y_test.shape[0],num_classes))
    for c in range(num_classes):
        y_test_pred[:,c] = models[c].test(X_test)
        y_p = (y_test_pred[:,c]>0.5)
        print("Class ",unique_classes[c]," Accuracy = ", sum(y_p==(y_test==unique_classes[c]))/(X_test.shape[0]))

    y_train_t = np.argmax(y_train_pred, axis=1)+1
    y_test_t = np.argmax(y_test_pred, axis=1)+1
    
    print("Train Accuracy : ",sum(y_train_t==y_train)/y_train.shape[0])
    print("Test Accuracy : ",sum(y_test_t==y_test)/y_test.shape[0])
    
    # Confusion Matrix
    conf_mat = np.ndarray((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            conf_mat[i][j] = sum((y_test_t==unique_classes[i]) & (y_test==unique_classes[j]))
    
    print(conf_mat)
        

    
    
    