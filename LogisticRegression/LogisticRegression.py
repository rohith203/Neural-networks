"""
This file contains the classes that implement Logistic Regression
"""
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import NormalScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

class LogisticRegression:
    """
    This class implements Logistic Regression using both
    Batch Gradient Descent and Stochastic Gradient descent.
    Class attributes:
		W  	     : current set of weights
		W_arr    : list of weights at each iteration
		Cost	 : current cost
		cost_arr : list of costs at each iteration
    """
    def init_weights(self, s):
        """
        This method initializes the weight matrix
        as a column vector with shape = (X rows+1, 1)
        """
        np.random.seed(11)
        self.W = np.random.randn(s[1],1)
        self.W_arr = []
        self.cost_arr = []
        self.cost = 0
        self.gradient = 0
        
    def sigmoid(self,x):
        """
        Sigmoid activation function. 
        returns the sigmoid of the given input.
        """
        return 1/(1+np.exp(-x))
        
    def get_cost(self, X, y, W):
        """
        This function returns the cost with the given set of weights
        using the formula
        	J =1/2 ∑_(i=0)^m▒(h_ω (x^i )-y^i )^2         
        """
        total_cost = 0
        for i in range(X.shape[0]):
            total_cost += y[i]*np.log(self.get_h_i(X, i, W)) + (1-y[i])*np.log(1-self.get_h_i(X, i, W))
        return (0.5/X.shape[0])*total_cost
    
    def add_bias(self, X):
        """
        This function adds bias (a column of ones) to the feature vector X.
        """
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X), axis=1)
    
    def get_h_i(self, X, i, W):
        """
        This function returns the hypothesis of ith feature vector
        with the given weights W.
			h_w (x^i )=∑_(j=0)^n▒〖w_j 〖x^i〗_j 〗=x^i w
        """
        h_i = np.matmul(X[i].reshape(1,-1),W)
        return self.sigmoid(h_i[0][0])

    def batch_grad_descent(self, X, y, alpha, max_iter):
        """
        This function implements the Batch Gradient Descent algorithm.
        It runs for multiple iterations until either the weights converge or 
        iterations reach max_iter. At each iteration the weights are updated using 
        the following rule
            repeat until convergence{
                w_j^(t+1)=w_j^t-α∑_(i=1)^m▒〖(h_w (x^i )-y^i ) x_j^i 〗
            }            
        """
        W_new = self.W.copy()
        for _ in range(max_iter):
            grad = np.zeros((X.shape[0],1))
            for i in range(X.shape[0]):
                grad[i] = (-y[i]*(1-self.get_h_i(X, i, self.W)) + (1-y[i])*self.get_h_i(X, i, self.W))
            for j in range(X.shape[1]):
                W_new[j][0] = self.W[j][0] - (alpha/X.shape[0])*np.sum(grad*X[:,j:j+1].reshape(-1,1))
            self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
            if len(self.W_arr)>1:
                if sum(abs(self.W_arr[-2]-self.W_arr[-1]))<0.0001:
                    break
        return W_new
    
    def stochastic_grad_descent(self, X, y, alpha, max_iter):
        """
        This function implements the Stochastic Gradient Descent algorithm.
        It runs for multiple iterations until either the weights converge or 
        iterations reach max_iter. Weights are updated for every row of the 
        training set.

            repeat until convergence{
                randomly shuffle the feature matrix rows
                for each feature vector x^i {
                    update all weights j -> 0 to n+1
                    w_j^(t+1)=w_j^t-α(h_w (x^i )-y^i ) x_j^i
                }
            }            
        """
        mat = np.concatenate((X,y.reshape(-1,1)), axis=1)
        for _ in range(max_iter):
            W_new = self.W.copy()
            np.random.shuffle(mat)
            X = mat[:,0:3]
            y = mat[:,3]
            for i in range(X.shape[0]):
                grad = (-y[i]*(1-self.get_h_i(X, i, self.W)) + (1-y[i])*self.get_h_i(X, i, self.W))
                for j in range(X.shape[1]):
                    W_new[j][0] = self.W[j][0] - (alpha)*(grad[0]*X[i,j])
                self.W = W_new.copy()
            self.cost_arr.append(self.get_cost(X, y, self.W))
            self.W_arr.append(self.W)
            if len(self.W_arr)>1:
                if sum(abs(self.W_arr[-2]-self.W_arr[-1]))<0.0001:
                    break
        return self.W
                              
        
    def train(self, X, y, alpha, max_iter=100, option="batch"):
        """
        This function initiates the training process.
        It runs batch gradient descent by default and can also run 
        Stochastic gradient descent if the argument is passed.
        
        returns the cost list which has costs at every training iteration.
        """
        X = self.add_bias(X)
        self.init_weights(X.shape)
        if option=="batch":
            self.batch_grad_descent(X,y,alpha,max_iter)
        elif option=="stochastic":
            self.stochastic_grad_descent(X,y,alpha,max_iter)
        self.cost = self.cost_arr[len(self.cost_arr)-1]
        return self.cost_arr
        
    def test(self,X,W=""):
        """
        This function takes a feature matrix as test data and 
        predicts the target values using the trained weights.

        returns the predicted target values.
        """
        if W=="":W = self.W

        X = self.add_bias(X)
        y_pred = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.get_h_i(X,i,W)
            for j in range(X.shape[1]):
                y_pred[i] += X[i][j]*W[j][0]
            y_pred[i] = self.sigmoid(y_pred[i])
        return y_pred

if __name__ == "__main__":
    model = LogisticRegression()

    # data input
    data = pd.read_excel("./data3.xlsx",header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    X = data[[0,1,2,3]]
    y = data[4]-1
    
    # data preprocessing (Normal scaling)
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X.loc[:,j])
        X.loc[:,j] = mscaler.transform(X.loc[:,j])
    
    # holdout cross validation split
    train_percent = 0.6
    X_train = X[:int(train_percent*X.shape[0])]
    y_train = y[:int(train_percent*X.shape[0])]
    X_test = X[int(train_percent*X.shape[0]):]
    y_test = y[int(train_percent*X.shape[0]):]
    
    # Training the model by choosing alpha and max_iter values.
	# gradient descent algorithm can be set as either ‘batch’ or ‘stochastic’
	# in this function call.
    alpha = 0.26
    max_iter = 100
    algo = 'batch'
    model.train(X_train.values,y_train.values,alpha,max_iter,algo)   

# Testing on train set    
    print("\nTraining..")
    y_pred = model.test(X_train.values)
    for i in range(y_pred.shape[0]):
        y_pred[i] = 0 if y_pred[i]<0.5 else 1

    print('\n',y_pred)
    print("\nTraining set accuracy: ",sum(y_pred==y_train)/y_train.shape[0])
    print("Training set sensitivity: ",sum((y_pred==1) & (y_train==1))/sum(y_train==1))
    print("Training set specificity: ",sum((y_pred==0) & (y_train==0))/sum(y_train==0))

# Testing on test set
    print("\nTesting...")
    y_pred = model.test(X_test.values)
    for i in range(y_pred.shape[0]):
        y_pred[i] = 0 if y_pred[i]<0.5 else 1

    print('\n',y_pred)
    print("\nTesting set accuracy: ",sum(y_pred==y_test)/y_test.shape[0])
    print("Training set sensitivity: ",sum(y_pred*y_test)/sum(y_test))
    print("Training set specificity: ",sum((y_pred==0) & (y_test==0))/sum(y_test==0))