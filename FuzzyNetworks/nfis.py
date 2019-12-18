import numpy as np
import matplotlib.pyplot as plt
from preprocessing import NormalScaler
import pandas as pd

class Network:
    def __init__(self, X, y, n_rules):
        self.Layers = []
        m = X.shape[0]
        n = X.shape[1]
        k = y.shape[1]
        self.att = {
        # input layer
        'in': np.ndarray((n,m)),
        # Layer 1 (Membership Layer)
        'mu': np.ndarray(shape = (n_rules, n, m)),
        'c': np.random.randn(n_rules, n),
        # 'c': np.random.uniform(low=-1, high=1 , size=(n_rules, n)),
        'c_err': np.zeros(shape=(n_rules, n)),
        'sigma': np.random.rand(n_rules, n),
        'sigma_err': np.zeros(shape = (n_rules, n)),
        
        # Layer 2 (Power Layer)
        'alpha': np.random.randn(n_rules, n, m),
        'p': np.random.uniform(low=0.1, high=4, size=(n_rules, n)),
        'p_err': np.zeros(shape = (n_rules, n)),

        # Layer 3 (Fuzzification Layer)
        'beta': np.random.randn(n_rules, m),

        # Layer 4 (De-fuzzification Layer)
        'o': np.random.randn(m, k),
        'w':np.random.randn(n_rules, k),
        'b':np.random.randn(),
        'w_err':np.zeros(shape=(n_rules, k)),
        'b_err':0,

        # Layer 5 (Normalization Layer)
        'h':np.ndarray(shape = (m, k)),
        'delta':np.ndarray(shape = (m, 1)),

        }
        # self.att['c'] = np.mean(X_train, axis=0).reshape(1,-1) * np.ones((n_rules, n))
        # self.att['mu'] = np.random.uniform(low=0.001, high=1, size=(n_rules,n,m))


    def print_shapes(self):
        for k,v in self.att.items():
            if k!='b' and k!='b_err':
                print(f'{k}: {v.shape} {v}')

    def feed_forward(self, X):
        '''
        In this function the given data set samples are propogated
        forward in the neural network.
        '''
        self.att['in'] = X[j].reshape(-1,1)
        for i in range(X.shape[0]):
            self.att['c'] = X[i].reshape(1,-1) - self.att['sigma']*np.sqrt(abs(np.log(self.att['mu'][:,:,i])))
            self.att['mu'][:,:,i] = np.exp(-0.5 * np.square((X[i].reshape(1,-1) - self.att['c'])/self.att['sigma']))
            self.att['alpha'][:,:,i] = np.power(self.att['mu'][:,:,i], self.att['p'])
            self.att['beta'][:,i] = np.product(self.att['alpha'][:,:,i], axis=1)
        self.att['o'] = self.att['beta'].T @ self.att['w']        
        self.att['delta'] = np.sum(self.att['o'], axis=1).reshape(-1,1)
        for i in range(X.shape[0]):
            self.att['h'][i,:] = (self.att['o'][i]/self.att['delta'][i]).reshape(1,-1)

        # self.print_shapes()
        return self.att['h']

    def train(self, X, y, X_test, y_test, lr, batch_size, max_iter):
        '''
        This function takes the training data and target values,
        applies forward propogation, then applies backward propogation
        to update the paramater matrices.
        batch gradient descent has been used to update weights.
        '''
        m = y.shape[0]
        k = y.shape[1]
        n = X.shape[1]
        n_rules = self.att['c'].shape[0]

        for iteration in range(max_iter):
            self.att['c_err'].fill(0)
            self.att['p_err'].fill(0)
            self.att['sigma_err'].fill(0)
            self.att['w_err'].fill(0)
            self.att['b_err'] = 0

            self.feed_forward(X)
            temp = (self.att['h'] - y) * ((1-self.att['h'])/(self.att['delta']))
            temp = (self.att['w'] @ (self.att['beta'] @ temp).T)

            for i in range(X.shape[0]):
                self.att['c_err'] += (temp  @ self.att['p'])* (X[i].reshape(1,-1) - self.att['c'])/(np.square(self.att['sigma']))
                self.att['sigma_err'] += (temp @ self.att['p']) * np.square(X[i].reshape(1,-1) - self.att['c'])/((self.att['sigma'])**3)

            self.att['p_err'] = temp @ np.sum(np.log(self.att['mu']), axis=2)

            self.att['w_err'] += self.att['beta'] @ ((self.att['h'] - y) \
                                * (self.att['delta'] - self.att['o'])/(np.square(self.att['delta']))) 

            self.att['c'] -= (lr/(m*k))*self.att['c_err']
            self.att['p'] -= (lr/(m*k))*self.att['p_err']
            self.att['sigma'] -= (lr/(m*k))*self.att['sigma_err']
            self.att['w'] -= (lr/(m*k))*self.att['w_err']

            cost = self.get_cost(X,y)
            print(f"iteration {iteration+1}  cost = {cost}")

    def get_cost(self, X, y):
        cost = 0
        # forward propogation
        self.feed_forward(X)
        cost = np.sum((self.att['h']-y)**2)
        return cost/(2*X.shape[0]*y.shape[1])

    def predict(self, X_test):
        self.feed_forward(X_test)
        return self.att['h']


if __name__ == "__main__":
    # data input
    data = pd.read_excel("./data4.xlsx",header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.values

    X = data[:, :7]
    y = data[:,7]

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    # data preprocessing
    mscaler = NormalScaler()
    for j in range(X.shape[1]):
        mscaler.fit(X[j])
        X[j] = mscaler.transform(X[j])
    
    y_cat = (y==unique_classes[0]).astype('int').reshape(-1,1)
    for i in unique_classes[1:]:
        y_cat = np.concatenate((y_cat,(y==i).astype('int').reshape(-1,1)),axis=1)

    # splitting data using holdout cross validation
    train_percent = 0.7
    X_train = X[:int(train_percent*X.shape[0])]
    y_train = y[:int(train_percent*X.shape[0])]
    y_cat_train = y_cat[:int(train_percent*X.shape[0])]
    X_test = X[int(train_percent*X.shape[0]):]
    y_test = y[int(train_percent*X.shape[0]):]
    y_cat_test = y_cat[int(train_percent*X.shape[0]):]

    alpha = 1e-6
    batch_size = 16
    max_iter = 30

    model = Network(X_train, y_cat_train, 10)
    model.train(X_train, y_cat_train, X_test, y_cat_test, alpha, batch_size, max_iter)
    # model.feed_forward(X_train)
    # print("COST: ",model.get_cost(X_train, y_cat_train))