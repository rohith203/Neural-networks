import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from preprocessing import NormalScaler

class LinearRegression_Vectorized:
    """
    This class implements Linear Regression using both
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
    
    def get_cost(self, X, y, W):
        """
        This function returns the cost with the given set of weights
        using the formula
        	J =1/2 ∑_(i=0)^m▒(h_ω (x^i )-y^i )^2         
        """
        return (0.5/X.shape[0])*np.sum(np.square(np.matmul(X,W)-y.reshape(-1,1)))

    def add_bias(self, X):
        """
        This function adds bias (a column of ones) to the feature vector X.
        """
        bias = np.ones((X.shape[0],1))
        return np.concatenate((bias,X), axis=1)
                                
    def train(self, X, y):
        """
        This function uses the vectorized version of linear regression
        and obtains the optimal weights with the given feature matrix
        and target values.
            todo: formula
        returns the optimal weights.
        """
        X = self.add_bias(X)
        self.init_weights(X.shape)
        self.W = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), y.reshape(-1,1))
        self.cost = self.get_cost(X,y,self.W)
        return self.W
    
    def test(self,X):
        """
        This function takes a feature matrix as test data and 
        predicts the target values using the trained weights.

        returns the predicted target values.
        """
        X = self.add_bias(X)
        return np.matmul(X,self.W)

if __name__ == "__main__":
    model = LinearRegression_Vectorized()
    
    # data input
    data = pd.read_excel("./data.xlsx", header=None)
    X = data.loc[:,0:1].values
    y = data.loc[:,2].values

    # data preprocessing (Normal Scaling)
    print(X[:,0].mean())
    mscaler = NormalScaler()
    mscaler.fit(X[:,0])
    X[:,0] = mscaler.transform(X[:,0])
    mscaler.fit(X[:,1])
    X[:,1] = mscaler.transform(X[:,1])

    # training the model
    arr = model.train(X,y)
    print("weights: ",model.W)
    print("Total Cost: ",model.cost)


    # visualization of cost function.
    res = 100
    bounds = 10
    xx = np.linspace(model.W[1]-bounds, model.W[1]+bounds, res)
    yy = np.linspace(model.W[2]-bounds, model.W[2]+bounds, res)
    minw0 = model.W[0]

    r = np.ndarray((res,res))
    s = np.ndarray((res,res))
    z = np.ndarray((res,res))

    for i in range(res):
        for j in range(res):
            z[i][j] = model.get_cost(model.add_bias(X), y, np.array([minw0,xx[i],yy[j]]).reshape(-1,1))
            r[i][j] = xx[i]
            s[i][j] = yy[j]

    # 3d surface plot of cost function and learning curve
    ax = plt.axes(projection='3d')
    ax.plot_surface(r, s, z,cmap='coolwarm')
    ax.text2D(0.05, 0.95, "3D surface plot of cost function", transform=ax.transAxes)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("cost")
    plt.savefig("./Results/lin_reg_vec/surf.png")
    plt.show()