# -*- coding: utf-8 -*-
"""
==============================================================================
Created on Dec 11 2019
Author: Yaxiong Yuan
==============================================================================
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from callbacks_custom import callbacks_custom

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers, optimizers

from optimizers_custom.ProxSGD_optimizer import ProxSGD
from optimizers_custom.ProxSGD_BNN_optimizer import ProxSGD_BNN
from optimizers_custom.ADABound_optimizer import ADABound
from optimizers_custom.AMSGrad_optimizer import AMSGrad

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train[0:60000, :, :, :]
    y_train = y_train[0:60000]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)
    return X_train, Y_train, X_test, Y_test
 
def create_model(mu_reg=1e-4, active_bnn=1):
    """
    create_model:
    # Arguments
        mu_reg: decides the regularization item
        active_bnn(1 or 0): whether to apply BNN
    """
    inputs = Input(shape=(28, 28, 1))
    layer_out = Flatten()(inputs)

    if active_bnn == 1:
        layer_out = Dense(200, name='dense_1', activation='tanh')(layer_out)
        layer_out = Dense(200, name='dense_2', activation='tanh')(layer_out)
        layer_out = Dense(200, name='dense_3', activation='tanh')(layer_out)
        layer_out = Dense(200, name='dense_4', activation='tanh')(layer_out)
        layer_out = Dense(200, name='dense_5', activation='tanh')(layer_out)
        layer_out = Dense(10, activation='softmax', name='Predictions')(layer_out)
    else:
        layer_out = Dense(200, activation='relu', name='dense_1', kernel_regularizer=regularizers.l1(mu_reg))(layer_out)
        layer_out = Dense(200, activation='relu', name='dense_2', kernel_regularizer=regularizers.l1(mu_reg))(layer_out)
        layer_out = Dense(200, activation='relu', name='dense_3', kernel_regularizer=regularizers.l1(mu_reg))(layer_out)
        layer_out = Dense(200, activation='relu', name='dense_4', kernel_regularizer=regularizers.l1(mu_reg))(layer_out)
        layer_out = Dense(200, activation='relu', name='dense_5', kernel_regularizer=regularizers.l1(mu_reg))(layer_out)
        layer_out = Dense(10, activation='softmax', name='Predictions', kernel_regularizer=regularizers.l1(mu_reg))(layer_out)

    model = Model(inputs=inputs, outputs=layer_out)
    return model

def train_model(mu_reg=1e-4, active_bnn=1, optimtype='proxsgd_bnn', verbose=True, Epoch=5, T=1):
    """
    train_model:
    # Arguments
        mu_reg: regularization parameter
        optimtype: choose a optimizer ('proxsgd', 'proxsgd_bnn', 'adam', 'ams', 'adabound')
        verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = callback.
        T: Number of experimental cycles
    """
           
    if optimtype == 'proxsgd':
        optim = ProxSGD(epsilon_initial=0.06, epsilon_decay=0.5, rho_initial=0.9, rho_decay=0.5, beta=0.999, mu=5e-5,
                        clip_bounds=None)
        model = create_model(mu_reg, active_bnn)

    elif optimtype == 'proxsgd_bnn':
        optim = ProxSGD_BNN(epsilon_initial=0.06, epsilon_decay=0.5, rho_initial=0.9, rho_decay=0.5, beta=0.999, mu_x=None,
                            clip_bounds_x=[-1.0, 1.0], mu_a=None, clip_bounds_a=[0.0, 1.0])
        model = create_model(mu_reg, active_bnn)

    elif optimtype == 'ams':
        optim = AMSGrad(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model = create_model(mu_reg, active_bnn)

    elif optimtype == 'adabound':
        optim = ADABound(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        model = create_model(mu_reg, active_bnn)

    else:
        optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,)
        model = create_model(mu_reg, active_bnn)
        
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    X_train, Y_train, X_test, Y_test = load_data()
    if verbose:
        cb = [callbacks_custom.metrics(validation_data=(X_test, Y_test), training_data=(X_train, Y_train), Times=T)]
    else:
        cb = []
    batch_size = 90
    model.fit(X_train, Y_train, validation_split=0.1, shuffle='batch', batch_size=batch_size, epochs=Epoch, verbose=verbose, callbacks=cb)
    model.evaluate(X_test, Y_test, verbose=0)

if __name__ == "__main__":
    Round = 1
    Epoch = 20
    Loss_matrix = np.zeros([Round, Epoch])
    Accuracy_matrix = np.zeros([Round, Epoch])
    NormL1_matrix = np.zeros([Round, Epoch])
    Accuracy_train_matrix = np.zeros([Round, Epoch])
    Loss_train_matrix = np.zeros([Round, Epoch])
    for t in range(Round):
        train_model(mu_reg=1e-4, active_bnn=1, optimtype='proxsgd_bnn', verbose=1, Epoch=Epoch, T=t)

        Loss_proxsgd = pickle.load(open('./variables/Loss_proxsgd', 'rb'))
        Loss_matrix[t, :] = Loss_proxsgd
        Accuracy_proxsgd = pickle.load(open('./variables/Accuracy_proxsgd', 'rb'))
        Accuracy_matrix[t, :] = Accuracy_proxsgd
        Loss_proxsgd_train = pickle.load(open('./variables/Loss_proxsgd_train', 'rb'))
        Loss_train_matrix[t, :] = Loss_proxsgd_train
        Accuracy_proxsgd_train = pickle.load(open('./variables/Accuracy_proxsgd_train', 'rb'))
        Accuracy_train_matrix[t, :] = Accuracy_proxsgd_train
        NormL1_proxsgd = pickle.load(open('./variables/NormL1_proxsgd', 'rb'))
        NormL1_matrix[t, :] = NormL1_proxsgd

    pickle.dump(Loss_matrix, open('./variables/Loss_table', 'wb'))
    pickle.dump(Accuracy_matrix, open('./variables/Accuracy_table', 'wb'))
    pickle.dump(Loss_train_matrix, open('./variables/Loss_train_table', 'wb'))
    pickle.dump(Accuracy_train_matrix, open('./variables/Accuracy_train_table', 'wb'))
    pickle.dump(NormL1_matrix, open('./variables/NormL1_table', 'wb'))


    if os.path.isdir('./figures'):
        pass
    else:
        os.mkdir('./figures')

    Loss_values = pickle.load(open('./variables/Loss_train_table', 'rb'))
    Mean_loss_values = np.mean(Loss_values, axis=0)
    ep = np.arange(Epoch)
    plt.figure(1)
    plt.plot(ep, Mean_loss_values)
    plt.grid(True)
    plt.savefig('./figures/mnist_loss.png')
    plt.show()

    Accuracy_values = pickle.load(open('./variables/Accuracy_table', 'rb'))
    Mean_accuracy_values = np.mean(Accuracy_values, axis=0)
    ep = np.arange(Epoch)
    plt.figure(2)
    plt.plot(ep, Mean_accuracy_values)
    plt.grid(True)
    plt.savefig('./figures/mnist_acc.png')
    plt.show()

    Weights_values = pickle.load(open('./variables/Weights_proxsgd', 'rb'))
    data = Weights_values
    data_flattened = np.concatenate((data[0].flatten(), data[1].flatten(), data[2].flatten(),
                                     data[3].flatten(), data[4].flatten(), data[5].flatten()), axis=None)
    x = np.sort(data_flattened)
    y = np.arange(1, len(x)+1)/len(x)
    plt.figure(3)
    plt.plot(x, y)
    plt.xlim((-3, 3))
    plt.grid(True)
    plt.savefig('./figures/mnist_cdf.png')
    plt.show()
