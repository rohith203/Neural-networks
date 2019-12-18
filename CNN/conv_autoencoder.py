import numpy as np
from preprocessing import NormalScaler
from scipy.io import loadmat
from keras.models import Model, Sequential
from keras.layers import Dense, Conv1D, Flatten, Lambda, MaxPooling1D, UpSampling1D, Conv2DTranspose, Input,Reshape
from keras.engine.topology import Layer
import keras.backend as K
import keras

class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        # print("build", input_shape)
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        # self._model.summary()
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

if __name__=='__main__':
    # data input
    data = loadmat('./data_for_cnn.mat')['ecg_in_window']

    np.random.shuffle(data)

    # data preprocessing
    scaler = NormalScaler()
    for j in range(data.shape[1]):
        scaler.fit(data[:,j])
        data[:,j] = scaler.transform(data[:,j])

    # holdout split
    split_percent = 0.7

    X_train = data[:int(data.shape[0]*split_percent), :].astype(np.float)
    X_test = data[int(data.shape[0]*split_percent): , :].astype(np.float)

    X_train = X_train.reshape(X_train.shape[0], 1000, 1)
    X_test = X_test.reshape(X_test.shape[0], 1000, 1)

    # number of filters
    filters = 10

    # Encoder
    inp = Input(shape=(1000,1))
    l1 = Conv1D(filters, 10, strides=2, activation='relu')(inp)
    l2 = MaxPooling1D(2)(l1)

    l3 = Flatten()(l2)
    l4 = Dense(248*filters, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(l3)

    # Decoder
    l4 = Reshape((248, filters)) (l4)
    l5 = UpSampling1D(2)(l4)
    out = Conv1DTranspose(1, 10, strides=2)(l5)

    model = Model(inp, out)

    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(X_train, X_train, validation_data=(X_test,X_test) , batch_size=500, epochs=200)

    # Results visualization
    plt.figure()
    plt.title(f'mse vs iteration plot')
    plt.xlabel("iteration")
    plt.ylabel("mse")
    plt.legend(loc='upper right')
    plt.plot(hist.history['val_loss'],c='r',label='validation set loss')
    plt.plot(hist.history['loss'],c='b',label='training set loss')