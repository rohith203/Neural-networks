import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from LogisticRegression import LogisticRegression,NormalScaler

def predictOneVsAll(X_train, y_train, X_test, y_test, unique_classes):
    num_classes = len(unique_classes)
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
    
    test_acc = sum(y_test_t==y_test)/y_test.shape[0]
    print("Train Accuracy : ",sum(y_train_t==y_train)/y_train.shape[0])
    print("Test Accuracy : ",test_acc)
    
    # Confusion Matrix
    conf_mat = np.ndarray((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            conf_mat[i][j] = sum((y_test_t==unique_classes[i]) & (y_test==unique_classes[j]))
    
    print(conf_mat)
    return test_acc

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
    
    k = 5
    N = X.shape[0]
    j = 0
    acc = 0
    for i in range(0,k):
        X_train = np.concatenate((X[:i*(N//k)],X[(i+1)*(N//k):]))
        y_train = np.concatenate((y[:i*(N//k)],y[(i+1)*(N//k):]))
        y_cat_train = np.concatenate((y_cat[:i*(N//k)],y_cat[(i+1)*(N//k):]))
        X_test = X[i*(N//k):(i+1)*(N//k)]
        y_test = y[i*(N//k):(i+1)*(N//k)]
        y_cat_test = y_cat[i*(N//k):(i+1)*(N//k)]
        acc += predictOneVsAll(X_train, y_train, X_test, y_test, unique_classes)
    print("Average Accuracy: ", acc/k)
       