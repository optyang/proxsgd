# -*- coding: utf-8 -*-
"""
this .py file includes the proposed ProxSGD optimizer
"""

from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K

#----------------------------------------------------------

class ProxSGD(Optimizer):
    """ProxSGD optimizer, proposed in
    ProxSGD: Training Structured Neural Networks under Regularization and Constraints, ICLR 2020
    # Arguments
        epsilon_initial: float >= 0. initial learning rate.
        epsilon_decay  : float >= 0. learning rate decay over each update.
        rho_initial    : float >= 0. initial step size for momentum.
        rho_decay      : float >= 0. momentum decay over each update.
        beta           : float >= 0. second momentum parameter.
        mu             : float >= 0. regularization parameter for L1 norm.
        clipping_bound : A vector including clipping lower bound and upper bound.
    """

    def __init__(self, epsilon_initial=0.06, epsilon_decay=0.5, rho_initial=0.9, rho_decay=0.5, beta=0.999,
                 mu=1e-4, clip_bounds=None, **kwargs):
        super(ProxSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations      = K.variable(0, dtype='int64', name='iterations')
            self.epsilon_initial = K.variable(epsilon_initial, name='epsilon_initial')
            self.epsilon_decay   = K.variable(epsilon_decay, name='epsilon_decay')
            self.rho_initial     = K.variable(rho_initial, name='rho_initial')
            self.rho_decay       = K.variable(rho_decay, name='rho_decay')
            self.beta            = K.variable(beta, name='beta')
            self.mu              = mu
            self.clip_bounds     = clip_bounds

    def get_updates(self, loss, params):
        grads        = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        iteration    = K.cast(self.iterations, K.dtype(self.epsilon_decay))
        epsilon      = self.epsilon_initial / ((iteration + 4) ** self.epsilon_decay) # the current lr for weights, see (8) of the paper
        rho          = self.rho_initial / ((iteration + 4) ** self.rho_decay) # the current lr for momentum, see (6) of the paper
        beta         = self.beta # the learning rate for the squared gradient, see Table I of the paper
        delta        = 1e-07 # see Table I of the paper
        
        if self.clip_bounds is not None:
            low = self.clip_bounds[0]
            up  = self.clip_bounds[1]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        rs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + vs + rs

        for x, g, v, r in zip(params, grads, vs, rs):

            '''update for weights x'''
            v_new = (1 - rho) * v + rho * g # (6) of the paper
            r_new = beta * r + (1 - beta) * K.square(g) # same update rule as ADAM, see Table I of the paper
            tau   = K.sqrt(r_new / (1 - beta ** (iteration + 1))) + delta # same update rule as ADAM, see Table I of the paper

            x_tmp = x - v_new / tau
            
            if self.mu is not None: # apply soft-thresholding due to L1 norm
                mu_normalized = self.mu / tau
                x_hat = K.maximum(x_tmp - mu_normalized, 0) - K.maximum(-x_tmp - mu_normalized, 0)
            else:
                x_hat = x_tmp
                
            if self.clip_bounds is not None: # apply the bound constraints
                x_hat = K.clip(x_hat, low, up)
                
            x_new = x + epsilon * (x_hat - x) # update the weights
            
            '''variable update'''
            self.updates.append(K.update(v, v_new))
            self.updates.append(K.update(r, r_new))
            self.updates.append(K.update(x, x_new))
        return self.updates