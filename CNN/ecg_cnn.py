from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, AveragePooling1D
import pandas as pd
import numpy as np
from preprocessing import NormalScaler
import matplotlib.pyplot as plt
import keras

# loading data
data = loadmat('./data_for_cnn.mat')['ecg_in_window'].astype(np.float64)
data_labels = loadmat('./class_label.mat')['label'].astype(np.int)

data = np.concatenate((data, data_labels), axis=1)

np.random.shuffle(data)

# data preprocessing
scaler = NormalScaler()
for j in range(data.shape[1]-1):
    scaler.fit(data[:,j])
    data[:,j] = scaler.transform(data[:,j])

# splitting data into train and test sets
split_percent = 0.8

X_train = data[:int(data.shape[0]*split_percent), :1000].astype(np.float)
y_train = data[:int(data.shape[0]*split_percent), 1000:1001]
X_test = data[int(data.shape[0]*split_percent): , :1000].astype(np.float)
y_test = data[int(data.shape[0]*split_percent): , 1000:1001]

X_train = X_train.reshape(X_train.shape[0], 1000, 1)
X_test = X_test.reshape(X_test.shape[0], 1000, 1)

# Convolutinal Neural Network model
model = Sequential()
model.add(Conv1D(100, 20, strides=1, input_shape=(1000,1)))
model.add(AveragePooling1D(2))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
model.add(Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.001), metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=500, epochs=1000)

model.evaluate(X_test, y_test)

from sklearn.metrics import confusion_matrix
pred = model.predict(X_test,batch_size=500)
conf_mat = confusion_matrix(y_test, (pred>=0.5).astype(np.int))
print(conf_mat)

# Results visualization
plt.figure()
plt.title(f'mse vs iteration plot')
plt.xlabel("iteration")
plt.ylabel("mse")
plt.plot(hist.history['loss'], c='b', label='training set loss')
plt.plot(hist.history['val_loss'], c='r', label='testing set loss')
plt.legend(loc='upper right')

plt.figure()
plt.title(f'accuracy vs iteration plot')
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.plot(hist.history['acc'][::10], c='b', label='training set accuracy')
plt.plot(hist.history['val_acc'][::10], c='r', label='testing set accuracy')
plt.legend(loc='upper left')