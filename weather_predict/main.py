import datetime

from matplotlib import pyplot

import tensorflow as tf

from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential

from pandas import read_csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

min_max_scaler = MinMaxScaler(feature_range = (-1, 1))


def init_tensorflow():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def normalize_data(df):
    return min_max_scaler.fit_transform(df.values)


def load_data(filename):
    dataset = read_csv(filename)
    df = dataset.dropna()
    df = df[['Temperature', 'Rainfall']]

    print('--- Dataset info ---')
    print(df.describe())
    print('--------------------')

    print('--- Normalized Data ---')
    X = normalize_data(df)
    print(X)
    print(X.shape)
    print('-----------------------')

    print('--- Y data ---')
    y = X[:, 0]
    print(y)
    print(y.shape)
    print('-------------')


    print('--- Train/Test split ---')
    test_size_var = 0.20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_var, random_state = 42)
    print('Train X', X_train.shape)
    print('Test X', X_test.shape)
    print('------------------------')


    # n = 3
    # Xtrain = []
    # ytrain = []

    # for i in range(n, len(X_train)):
    #     Xtrain.append(X_train[i - n: i, : X_train.shape[1]])
    #     ytrain.append(y_train[i])
    # print(X_train.shape[1])
    # print(ytrain)
    # plt.figure(1, figsize =(16, 6))
    # plt.plot(df.Rainfall)
    # plt.show(block = False)
    # plt.show()
    return X_train, X_test, y_train, y_test

def get_data_time_steps(X, y, time_steps = 3):

    Xtrain = []
    ytrain = []

    for i in range(time_steps, len(X)):
        Xtrain.append(X[i - time_steps: i, : X.shape[1]])
        ytrain.append(y[i])

    return np.array(Xtrain), np.array(ytrain)

def GRU_Model(X_train, y_train, X_test, y_test):
    m, n, l = X_train.shape
    model = Sequential()
    model.add(GRU(128, input_shape = (n, l)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.summary()


    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X_train, y_train, batch_size=8, epochs=2000, verbose = 1, callbacks=[tensorboard_callback])


    train_pred = model.predict(X_train)


    test_score = model.evaluate(X_test, y_test , verbose = 1)
    print('Test Score : %.2f MSE (%.2f RMSE)', (test_score, np.sqrt(test_score)))


    test_pred = model.predict(X_test)

    train_pred = np.c_[train_pred, np.zeros(train_pred.shape)]
    test_pred = np.c_[test_pred, np.zeros(test_pred.shape)]

    train_pred = min_max_scaler.inverse_transform(train_pred)[:, 0]
    test_pred = min_max_scaler.inverse_transform(test_pred)[:, 0]

    print(train_pred)
    print(test_pred)
    plt.figure(1, figsize =(16, 6))
    plt.plot(range(0, len(test_pred)), test_pred)
    plt.show()



def main():
    X_train, X_test, y_train, y_test = load_data('./input/temp-rain.csv')
    time_steps = 3
    X_train, y_train = get_data_time_steps(X_train, y_train, time_steps)
    X_test, y_test = get_data_time_steps(X_test, y_test, time_steps)
    print('--- Time Step Prediction ---')
    print('Time step lookup: ', time_steps)
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)
    print('---------------------------')

    GRU_Model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
