import numpy as np
import matplotlib.pyplot as plt
from preprocessing import NormalScaler
from sklearn.metrics import confusion_matrix
import pandas as pd

class Network:
    def __init__(self, X, y, n_rules):
        self.Layers = []
        m = X.shape[0]
        n = X.shape[1]
        k = y.shape[1]
        self.cost_arr = {'train':[], 'test':[]}
        self.acc_arr = {'train':[], 'test':[]}
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
        '''
        In this function the given data set samples are propogated
        forward in the neural network.
        '''
        self.att['in'] = X[j].reshape(-1,1)
        # self.att['c'] = self.att['in'].T - self.att['sigma']*np.sqrt(abs(np.log(self.att['mu'])))
        self.att['mu'] = np.exp(-0.5 * np.square((self.att['in'].T - self.att['c'])/(self.att['sigma'])))
        for i in range(self.att['mu'].shape[0]):
            for l in range(self.att['mu'].shape[1]):
                if (self.att['mu'][i,l])<1e-10 : self.att['mu'][i,l] = 1e-6
                # if (self.att['mu'][i,l])<1e-10 and (self.att['mu'][i,l])<=0: self.att['mu'][i,l] = -1e-5
                # elif (self.att['mu'][i,l])<1e-10 and (self.att['mu'][i,l])>=0: self.att['mu'][i,l] = 1e-5


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
        to update the paramater matrices.
        batch gradient descent has been used to update weights.
        '''
        m = y.shape[0]
        k = y.shape[0]
        # self.att['p'].fill(0.0)
        for iteration in range(max_iter):
            for i in range(0,m-batch_size+1,batch_size):
                self.att['c_err'].fill(0)
                self.att['p_err'].fill(0)
                self.att['sigma_err'].fill(0)
                self.att['w_err'].fill(0)
                self.att['b_err'] = 0

                for j in range(i,i+batch_size):
                    # forward propogation
                    self.feed_forward(X, j)

                    # Backpropogation of errors
                    temp = (self.att['h'] - y[j].reshape(1,-1)) * ((1-self.att['h'])/self.att['delta'])
                    temp = ((self.att['beta'] @ temp).T) 
                    temp = self.att['w'] @ temp

                    self.att['c_err'] += (temp @ self.att['p']) * (X[j].reshape(1,-1) - self.att['c'])/(np.square(self.att['sigma']))

                    self.att['p_err'] += temp @ np.log(abs(self.att['mu']))

                    self.att['sigma_err'] += (temp @ self.att['p']) * (np.square(X[j].reshape(1,-1) - self.att['c'])/((self.att['sigma'])**3))

                    self.att['w_err'] += self.att['beta'] @ ((self.att['h'] - y[j].reshape(1,-1)) \
                                        * (self.att['delta'] - self.att['o'])/(np.square(self.att['delta']))) 
                
                
                # updating parameters after backpropogating each batch
                self.att['c'] -= (lr/(batch_size*k))*self.att['c_err']
                self.att['p'] -= (lr/(batch_size*k))*self.att['p_err']
                self.att['sigma'] -= (lr/(batch_size*k))*self.att['sigma_err']
                self.att['w'] -= (lr/(batch_size*k))*self.att['w_err']

            tr = self.evaluate(X,y)
            te = self.evaluate(X_test,y_test)
            self.cost_arr['train'].append(tr['loss'])
            self.cost_arr['test'].append(te['loss'])
            self.acc_arr['train'].append(tr['acc'])
            self.acc_arr['test'].append(te['acc'])
            print(f"iteration {iteration+1}  train_loss={self.cost_arr['train'][-1]} test_loss={self.cost_arr['test'][-1]}")

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
        pred = model.predict(X_test)
        conf_mat = confusion_matrix(y_test, np.argmax(pred, axis=1))
        # print(conf_mat)
        return {'acc':acc/y.shape[0], 'loss':loss, 'conf_mat':conf_mat}

if __name__ == "__main__":
    # data input
    data = pd.read_excel("./data4.xlsx",header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.values

    X = data[:, :7]
    y = data[:,7] - 1

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

    alpha = 0.1
    batch_size = 16
    max_iter = 600
    n_rules = 32
    model = Network(X_train, y_cat_train, n_rules)
    model.train(X_train, y_cat_train, X_test, y_cat_test, alpha, batch_size, max_iter)
    
    print('train: ',model.evaluate(X_train,y_cat_train))
    print('test: ', model.evaluate(X_test,y_cat_test))
    # model.feed_forward(X_train, 0)
    # print("COST: ",model.get_cost(X_train, y_cat_train))
    model.evaluate(X_test, y_cat_test)
    # model.print_shapes()

    plt.figure()
    plt.title(f'Cost Function vs iteration plot alpha={alpha} max_iter={max_iter} batch_size={batch_size}\n n_rules={n_rules}')
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.plot(model.cost_arr['train'],c='c',label='training set avg cost')
    plt.plot(model.cost_arr['test'], c='r',label='testing set avg cost')
    plt.legend(loc='upper right')
    # plt.savefig(f"./results/{alpha}_{max_iter}_{batch_size}_{n_rules}_loss.png")
    # plt.show()
    
    plt.figure()
    plt.title(f'Accuracy vs iteration plot alpha={alpha} max_iter={max_iter}\n batch_size={batch_size} n_rules={n_rules}')
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.plot(model.acc_arr['train'],c='c',label='training set acc')
    plt.plot(model.acc_arr['test'], c='r',label='testing set acc')
    plt.legend(loc='upper left')
    # plt.savefig(f"./results/{alpha}_{max_iter}_{batch_size}_{n_rules}_acc.png")
    # plt.show()