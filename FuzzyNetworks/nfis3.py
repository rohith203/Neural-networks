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
        'in': np.ndarray((n,1)),
        # Layer 1 (Membership Layer)
        'mu': np.ndarray(shape = (n_rules, n)),
        'c': np.random.randn(n_rules, n),
        # 'c': np.random.uniform(low=-1, high=1 , size=(n_rules, n)),
        'c_err': np.zeros(shape=(n_rules, n)),
        'sigma': np.random.rand(n_rules, n),
        'sigma_err': np.zeros(shape = (n_rules, n)),
        
        # Layer 2 (Power Layer)
        'alpha': np.random.randn(n_rules, n),
        'p': np.random.uniform(low=0.1, high=4, size=(n_rules, n)),
        'p_err': np.zeros(shape = (n_rules, n)),

        # Layer 3 (Fuzzification Layer)
        'beta': np.random.randn(n_rules, 1),

        # Layer 4 (De-fuzzification Layer)
        'o': np.random.randn(1, k),
        'w':np.random.randn(n_rules, k),
        'b':np.random.randn(),
        'w_err':np.zeros(shape=(n_rules, k)),
        'b_err':0,

        # Layer 5 (Normalization Layer)
        'h':np.ndarray(shape = (1, k)),
        'delta': 1,

        }
        self.att['mu'] = np.random.uniform(low=0.001, high=1, size=(n_rules,n))


    def print_shapes(self):
        for k,v in self.att.items():
            if k!='b' and k!='b_err':
                print(f'{k}: {v.shape} {v}')

    def feed_forward(self, X, j):
        # self.att['p'].fill(1.0)

        self.att['in'] = X[j].reshape(-1,1)
        self.att['c'] = self.att['in'].T - self.att['sigma']*np.sqrt(abs(np.log(self.att['mu'])))
        self.att['mu'] = np.exp(-0.5 * np.square((self.att['in'].T - self.att['c'])/(self.att['sigma'])))
        self.att['alpha'] = np.power(self.att['mu'], self.att['p'])
        self.att['beta'] = np.product(self.att['alpha'], axis=1).reshape(-1,1)
        self.att['o'] = self.att['beta'].T @ self.att['w']
        self.att['delta'] = np.sum(self.att['o'])
        self.att['h'] = (self.att['o']/self.att['delta'])

        # self.print_shapes()
        return self.att['h']

    def train(self, X, y, X_test, y_test, lr, batch_size, max_iter):
        '''
        This function takes the training data and target values,
        applies forward propogation, then applies backward propogation
        to update the weight matrices.
        mini-batch gradient descent has been used to update weights.
        '''
        m = y.shape[0]
        k = y.shape[1]
        n = X.shape[1]
        n_rules = self.att['c'].shape[0]
        # self.att['p'].fill(0.0)
        for iteration in range(max_iter):
            for i in range(0,m-batch_size+1,batch_size):
                self.att['c_err'].fill(0)
                self.att['p_err'].fill(0)
                self.att['sigma_err'].fill(0)
                self.att['w_err'].fill(0)
                self.att['b_err'] = 0


                for a in range(n_rules):
                    for b in range(n):
                        for j in range(i,i+batch_size):
                            # forward propogation
                            self.feed_forward(X, j)
                            for q in range(k):
                                # Backpropogation of errors
                                temp = (self.att['h'][0,q] - y[j][q]) * ((1-self.att['h'][0,q])/self.att['delta'])
                                temp = (self.att['beta'][a,0] * temp) * self.att['w'][a,q] 

                                self.att['c_err'][a,b] += (temp * self.att['p'][a,b]) * (X[j][b] - self.att['c'][a,b])/(np.square(self.att['sigma'][a,b]))

                                self.att['p_err'][a,b] += temp * np.log(abs(self.att['mu'][a,b]))

                                self.att['sigma_err'][a,b] += (temp * self.att['p'][a,b]) * (np.square(X[j, b] - self.att['c'][a,b])/((self.att['sigma'][a,b])**3))

                                self.att['w_err'][a,q] += self.att['beta'][a,0] * ((self.att['h'][0,q] - y[j][q]) \
                                                    * (self.att['delta'] - self.att['o'][0,q])/(np.square(self.att['delta']))) 
            
                
                # updating parameters after backpropogating each batch
                self.att['c'] -= (lr/(batch_size*k))*self.att['c_err']
                self.att['p'] -= (lr/(batch_size*k))*self.att['p_err']
                self.att['sigma'] -= (lr/(batch_size*k))*self.att['sigma_err']
                self.att['w'] -= (lr/(batch_size*k))*self.att['w_err']

            cost = self.get_cost(X,y)
            print(f"iteration {iteration+1}  cost = {cost}")

    def get_cost(self, X, y):
        cost = 0
        for i in range(y.shape[0]):
            # forward propogation
            self.feed_forward(X, i)
            cost += np.sum((self.att['h']-y[i].reshape(1,-1))**2)
            # print(self.att['h'],y[i],cost)
        return cost/(2*X.shape[0]*y.shape[1])

    def predict(self, X):
        pred = np.ndarray((X.shape[0],3))
        for i in range(X.shape[0]):
            self.feed_forward(X, i)
            pred[i] = self.att['h']
        return pred
    
    def evaluate(self, X, y):
        acc = 0
        for i in range(y.shape[0]):
            self.feed_forward(X, i)
            if int(np.argmax(self.att['h']))==int(np.argmax(y[i])):
                acc+=1
        loss = self.get_cost(X, y)
        return {'acc':acc/y.shape[0], 'loss':loss}

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

    alpha = 3.9
    batch_size = X_train.shape[0]
    max_iter = 30

    model = Network(X_train, y_cat_train, 10)
    model.train(X_train, y_cat_train, X_test, y_cat_test, alpha, batch_size, max_iter)
    
    print('train: ',model.evaluate(X_train,y_cat_train))
    print('test: ', model.evaluate(X_test,y_cat_test))
    # model.feed_forward(X_train, 0)
    # print("COST: ",model.get_cost(X_train, y_cat_train))
    # model.print_shapes()