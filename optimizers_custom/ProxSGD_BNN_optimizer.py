# -*- coding: utf-8 -*-
"""
this .py file includes the proposed ProxSGD optimizer for BNN
"""

from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K
import tensorflow as tf
#----------------------------------------------------------

class ProxSGD_BNN(Optimizer):
    """ProxSGD_BNN optimizer.
    (The optimizer specializes in BNN)

    # Arguments
        epsilon_initial: float >= 0. initial learning rate for weight.
        epsilon_decay:   float >= 0. learning rate decay over each update, for weight.
        rho_initial:     float >= 0. initial learning rate for momentum.
        rho_decay:       float >= 0. learning rate decay over each update, for momentum.
        beta:            float >= 0. second momentum parameter.
        mu_x:            float >= 0. L1 regularization, for weight.
        clip_bounds_x:   A vector including clipping lower bound and upper bound for weights.
        mu_a:            float >= 0. L1 regularization, for a.
        clip_bounds_a:   A vector including clipping lower bound and upper bound for a.
    """

    def __init__(self, epsilon_initial=0.06, epsilon_decay=0.5, rho_initial=0.9, rho_decay=0.5, beta=0.999,
                 mu_x=None, clip_bounds_x=None, mu_a=None, clip_bounds_a=None, **kwargs):
        super(ProxSGD_BNN, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.epsilon_initial = K.variable(epsilon_initial, name='epsilon_initial')
            self.epsilon_decay = K.variable(epsilon_decay, name='epsilon_decay')
            self.rho_initial = K.variable(rho_initial, name='rho_initial')
            self.rho_decay = K.variable(rho_decay, name='rho_decay')
            self.beta = K.variable(beta, name='beta')
            self.mu_x = mu_x
            self.clip_bounds_x = clip_bounds_x
            self.mu_a = mu_a
            self.clip_bounds_a = clip_bounds_a

    def get_updates(self, loss, params):
        grads_x      = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        iteration    = K.cast(self.iterations, K.dtype(self.epsilon_decay))
        epsilon      = self.epsilon_initial / ((iteration + 4) ** self.epsilon_decay)
        rho          = self.rho_initial / ((iteration + 4) ** self.rho_decay)
        beta         = self.beta
        delta        = 1e-8
        mu_bnn       = 0.00025
        if self.clip_bounds_x is not None:
            low_x = self.clip_bounds_x[0]
            up_x  = self.clip_bounds_x[1]
        if self.clip_bounds_a is not None:
            low_a = self.clip_bounds_a[0]
            up_a  = self.clip_bounds_a[1]

        vs_x = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        rs_x = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs_a = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        rs_a = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        params_a = [K.ones(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + vs_x + rs_x + params_a + vs_a + rs_a

        for x, g, v_x, r_x, a, v_a, r_a in zip(params, grads_x, vs_x, rs_x, params_a, vs_a, rs_a):
            '''update for weights x'''
            v_x_new = (1 - rho) * v_x + rho * (g + mu_bnn/2 * (2*a + x - 1))
            r_x_new = beta * r_x + (1 - beta) * K.square(g + mu_bnn/2 * (2*a + x - 1))
            tau_x = K.sqrt(r_x_new / (1 - beta ** (iteration + 1))) + delta

            x_tmp = x - v_x_new / tau_x
            if self.mu_x is not None:
                mu_x_normalized = mu_x / tau_x
                x_hat = K.maximum(x_tmp - mu_x_normalized, 0) - K.maximum(-x_tmp - mu_x_normalized, 0)
            else:
                x_hat = x_tmp
            if self.clip_bounds_x is not None:
                x_hat = K.clip(x_hat, low_x, up_x)
            x_new = x + epsilon * (x_hat - x)
            '''If you need to manually process the weights'''
            #x_new = K.sign(x_new)
            #x_new = tf.cond(K.greater(iteration, 33748), lambda: K.clip(x_new, -1, 1), lambda:x_new)
            #x_new = tf.cond(K.greater(iteration, 33748), lambda: K.sign(x_new), lambda: x_new)

            '''update for parameter a'''
            v_a_new = (1 - rho) * v_a + rho * mu_bnn * x
            r_a_new = beta * r_a + (1 - beta) * K.square(mu_bnn * x)
            tau_a = K.sqrt(r_a_new / (1 - beta ** (iteration + 1))) + delta

            a_tmp = a - v_a_new / tau_a
            if self.mu_a is not None:
                mu_a_normalized = mu_a / tau_a
                a_hat = K.maximum(a_tmp - mu_a_normalized, 0) - K.maximum(-a_tmp - mu_a_normalized, 0)
            else:
                a_hat = a - v_a_new / tau_a
            if self.clip_bounds_a is not None:
                a_hat = K.clip(a_hat, low_a, up_a)
            a_new = a + epsilon * (a_hat - a)
            '''If you need to manually process the parameters a'''
            #a_new = K.clip(a_new, 0, 1)
            #a_new = tf.cond(K.greater(iteration, 4000), lambda: K.clip(a_new, 0, 1), lambda:a_t)
            #a_new = K.sign(a_new) * K.maximum(K.abs(a_new) - t * epsilon, 0)

            '''variable update'''
            self.updates.append(K.update(v_a, v_a_new))
            self.updates.append(K.update(r_a, r_a_new))
            self.updates.append(K.update(a, a_new))
            self.updates.append(K.update(v_x, v_x_new))
            self.updates.append(K.update(r_x, r_x_new))
            self.updates.append(K.update(x, x_new))
        return self.updates

