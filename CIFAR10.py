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
from callbacks_custom import callbacks_custom_CNN
from binary_tanh_custom.binary_ops import binary_tanh


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Activation, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from optimizers_custom.ProxSGD_optimizer import ProxSGD
from optimizers_custom.ProxSGD_BNN_optimizer import ProxSGD_BNN
from optimizers_custom.ADABound_optimizer import ADABound
from optimizers_custom.AMSGrad_optimizer import AMSGrad

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def create_model(active_bnn=0):
    num_classes = 10
    model = Sequential()
    if active_bnn == 1:
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=(32, 32, 3), name='conv1'))
        model.add(BatchNormalization())
        model.add(Activation(binary_tanh))
        model.add(Conv2D(32, (3, 3), padding='same', name='conv2'))
        model.add(BatchNormalization())
        model.add(Activation(binary_tanh))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (5, 5), padding='same', name='conv3'))
        model.add(BatchNormalization())
        model.add(Activation(binary_tanh))
        model.add(Conv2D(64, (3, 3), padding='same', name='conv4'))
        model.add(BatchNormalization())
        model.add(Activation(binary_tanh))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (3, 3), padding='same', name='conv5'))
        model.add(BatchNormalization())
        model.add(Activation(binary_tanh))
        model.add(Conv2D(128, (3, 3), padding='same', name='conv6'))
        model.add(BatchNormalization())
        model.add(Activation(binary_tanh))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', name='fc1'))
    else:
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), name='conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same', name='conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same', name='conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), padding='same', name='conv5'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same', name='conv6'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', name='fc1'))
    return model

def train_model(active_bnn=0, optimtype='proxsgd', verbose=True, Epoch=5, T=1):
    x_train, y_train, x_test, y_test = load_data()
    # data augmentation
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)

    if optimtype == 'proxsgd':
        optim = ProxSGD(epsilon_initial=0.06, epsilon_decay=0.5, rho_initial=0.9, rho_decay=0.5, beta=0.999, mu=1e-4,
                        clip_bounds=None)
        model = create_model(active_bnn)

    elif optimtype == 'proxsgd_bnn':
        optim = ProxSGD_BNN(epsilon_initial=0.05, epsilon_decay=0.6, rho_initial=0.9, rho_decay=0.6, beta=0.999, mu_x=None,
                            clip_bounds_x=[-1.0, 1.0], mu_a=None, clip_bounds_a=[0.0, 1.0])
        model = create_model(active_bnn)

    elif optimtype == 'ams':
        optim = AMSGrad(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model = create_model(active_bnn)

    elif optimtype == 'adabound':
        optim = ADABound(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        model = create_model(active_bnn)

    else:
        optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,)
        model = create_model(active_bnn)

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    if verbose:
        cb = [callbacks_custom_CNN.metrics(validation_data=(x_test, y_test), training_data=(x_train, y_train), Times=T)]
    else:
        cb = []
    batch_size = 64
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), \
                        steps_per_epoch=x_train.shape[0] // batch_size, epochs=Epoch, \
                        callbacks=cb, verbose=verbose, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test, verbose=0)

if __name__ == "__main__":
    Round = 1
    Epoch = 20
    Loss_matrix = np.zeros([Round, Epoch])
    Accuracy_matrix = np.zeros([Round, Epoch])
    NormL1_matrix = np.zeros([Round, Epoch])
    Accuracy_train_matrix = np.zeros([Round, Epoch])
    Loss_train_matrix = np.zeros([Round, Epoch])
    for t in range(Round):
        train_model(active_bnn=0, optimtype='proxsgd', verbose=1, Epoch=Epoch, T=t)

        Loss_proxsgd = pickle.load(open('./variables_cnn/Loss_proxsgd', 'rb'))
        Loss_matrix[t, :] = Loss_proxsgd
        Accuracy_proxsgd = pickle.load(open('./variables_cnn/Accuracy_proxsgd', 'rb'))
        Accuracy_matrix[t, :] = Accuracy_proxsgd
        Loss_proxsgd_train = pickle.load(open('./variables_cnn/Loss_proxsgd_train', 'rb'))
        Loss_train_matrix[t, :] = Loss_proxsgd_train
        Accuracy_proxsgd_train = pickle.load(open('./variables_cnn/Accuracy_proxsgd_train', 'rb'))
        Accuracy_train_matrix[t, :] = Accuracy_proxsgd_train
        NormL1_proxsgd = pickle.load(open('./variables_cnn/NormL1_proxsgd', 'rb'))
        NormL1_matrix[t, :] = NormL1_proxsgd

    pickle.dump(Loss_matrix, open('./variables_cnn/Loss_table', 'wb'))
    pickle.dump(Accuracy_matrix, open('./variables_cnn/Accuracy_table', 'wb'))
    pickle.dump(Loss_train_matrix, open('./variables_cnn/Loss_train_table', 'wb'))
    pickle.dump(Accuracy_train_matrix, open('./variables_cnn/Accuracy_train_table', 'wb'))
    pickle.dump(NormL1_matrix, open('./variables_cnn/NormL1_table', 'wb'))

    Weights_values = pickle.load(open('./variables_cnn/Weights_proxsgd', 'rb'))
    data = Weights_values
    data_flattened = np.concatenate((data[0].flatten(), data[1].flatten(), data[2].flatten(),
                                     data[3].flatten(), data[4].flatten(), data[5].flatten()), axis=None)

    if os.path.isdir('./figures'):
        pass
    else:
        os.mkdir('./figures')

    Loss_values = pickle.load(open('./variables_cnn/Loss_train_table', 'rb'))
    Mean_loss_values = np.mean(Loss_values, axis=0)
    ep = np.arange(Epoch)
    plt.figure(1)
    plt.plot(ep, Mean_loss_values)
    plt.savefig('./figures/cifar10_loss.png')
    plt.show()

    Accuracy_values = pickle.load(open('./variables_cnn/Accuracy_table', 'rb'))
    Mean_accuracy_values = np.mean(Accuracy_values, axis=0)
    ep = np.arange(Epoch)
    plt.figure(2)
    plt.plot(ep, Mean_accuracy_values)
    plt.savefig('./figures/cifar10_acc.png')
    plt.show()

    Weights_values = pickle.load(open('./variables_cnn/Weights_proxsgd', 'rb'))
    data = Weights_values
    data_flattened = np.concatenate((data[0].flatten(), data[1].flatten(), data[2].flatten(),
                                     data[3].flatten(), data[4].flatten(), data[5].flatten()), axis=None)
    x = np.sort(data_flattened)
    y = np.arange(1, len(x)+1)/len(x)
    plt.figure(3)
    plt.plot(x, y)
    plt.savefig('./figures/cifar10_cdf.png')
    plt.show()
