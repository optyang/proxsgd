"""
=============================================================================
    Eindhoven University of Technology
    Philips Research
==============================================================================

    Source Name   : callbacks_custom.py
                    callback fuction which displays weights 
                    at the end of each epoch

    Author        : Ruud van Sloun
    Date          : 08/09s/2018

==============================================================================
"""
import tensorflow.keras as keras
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt


class weights_vis(keras.callbacks.Callback):
    def __init__(self,every_nth_epoch=1, figsize=(12,8)):
        """Init callback
        Args:
            test_data : input test dataset (inputs,targets)
        Returns:
            void
        """        
        self.figsize = figsize
        self.every_nth_epoch = every_nth_epoch
        self.count = 0
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class metrics(keras.callbacks.Callback):
    def __init__(self, validation_data, training_data, Times = 0):
        """Init callback
        Args:
            test_data : input test dataset (inputs,targets)
        Returns:
            void
        """        

        self.X = validation_data[0]
        self.Y = validation_data[1]
        self.X_train = training_data[0]
        self.Y_train = training_data[1]
        self.batch = 0
        self.epoch = 0
        self.loss = []
        self.acc = []
        self.loss_train = []
        self.acc_train = []
        self.connects = []
        self.neurons = []
        self.l1 = []
        self.time = Times

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        '''YY: data to be saved is list, but it is converted to np array when being saved
        For example, the data to be saved is a list [1,2,3] of length 3 -> savemat -> loadmat: the data becomes ny array [[1,2,3]] of length 1
        '''

        if os.path.isdir('./variables'):
            pass
        else:
            os.mkdir('./variables')

        pickle.dump(self.loss, open('./variables/Loss_proxsgd', 'wb'))
        pickle.dump(self.acc, open('./variables/Accuracy_proxsgd', 'wb'))
        pickle.dump(self.loss_train, open('./variables/Loss_proxsgd_train', 'wb'))
        pickle.dump(self.acc_train, open('./variables/Accuracy_proxsgd_train', 'wb'))
        pickle.dump(self.l1, open('./variables/NormL1_proxsgd', 'wb'))
        pickle.dump(self.w_net, open('./variables/Weights_proxsgd', 'wb'))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = self.epoch+1
        score = self.model.evaluate(self.X, self.Y, verbose=0)
        print(score)
        self.loss.append(score[0])
        self.acc.append(score[1])
        score_train = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        self.loss_train.append(score_train[0])
        self.acc_train.append(score_train[1])
        self.w_net = []
        b_net = []
        w,b = self.model.get_layer('dense_1').get_weights()
        self.w_net.append(w)
        b_net.append(b)
        w,b = self.model.get_layer('dense_2').get_weights()
        self.w_net.append(w)
        b_net.append(b)
        w, b = self.model.get_layer('dense_3').get_weights()
        self.w_net.append(w)
        b_net.append(b)
        w, b = self.model.get_layer('dense_4').get_weights()
        self.w_net.append(w)
        b_net.append(b)
        w, b = self.model.get_layer('dense_5').get_weights()
        self.w_net.append(w)
        b_net.append(b)
        w, b = self.model.get_layer('Predictions').get_weights()
        self.w_net.append(w)
        b_net.append(b)
        l1_norm = 0
        for weights_layer in self.w_net:
                l1_norm += np.sum(np.absolute(weights_layer))
        self.l1.append(l1_norm)

        #-------------------------------------------------------------------------------------------------
        if os.path.isdir('./figures'):
            pass
        else:
            os.mkdir('./figures')
        
        Weights_values = self.w_net
        data = Weights_values
        data_flattened = np.concatenate((data[0].flatten(), data[1].flatten(), data[2].flatten(),
                                         data[3].flatten(), data[4].flatten(), data[5].flatten()), axis=None)
        x = np.sort(data_flattened)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.figure(4)
        plt.plot(x, y)
        plt.xlim((-3, 3))
        plt.grid(True)
        plt.savefig('./figures/mnist_cdf_temp.png')
        plt.show()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
