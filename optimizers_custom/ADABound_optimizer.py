# -*- coding: utf-8 -*-
"""
this .py file includes ADABound optimizer
"""

from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K
import tensorflow as tf

#----------------------------------------------------------
class ADABound(Optimizer):
    """Adabound.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".

    # References
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., **kwargs):
      super(ADABound, self).__init__(**kwargs)
      with K.name_scope(self.__class__.__name__):
        self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.beta_1 = K.variable(beta_1, name='beta_1')
        self.beta_2 = K.variable(beta_2, name='beta_2')
        self.decay = K.variable(decay, name='decay')
      if epsilon is None:
        epsilon = K.epsilon()
      self.epsilon = epsilon
      self.initial_decay = decay

    def get_updates(self, loss, params):
      grads = self.get_gradients(loss, params)
      self.updates = [K.update_add(self.iterations, 1)]

      lr = self.lr
      if self.initial_decay > 0:
        lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

      t = K.cast(self.iterations, K.floatx()) + 1
      lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                   (1. - K.pow(self.beta_1, t)))
      # lr_t = lr_t / K.sqrt(t)

      etal_t = 0.1 - 0.1 / K.pow((1 - self.beta_2), t + 1)
      etau_t = 0.1 + 0.1 / K.pow((1 - self.beta_2), t)

      # etal_t = 0.1 - 0.1 / ((1-self.beta_2) * t + 1)
      # etau_t = 0.1 + 0.1 / ((1-self.beta_2) * t)

      ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
      vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

      self.weights = [self.iterations] + ms + vs

      for p, g, m, v in zip(params, grads, ms, vs):
        m_t = (self.beta_1 / t * m) + (1. - self.beta_1 / t) * g
        v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

        eta = lr_t / (K.sqrt(v_t) + self.epsilon)
        eta_t = tf.clip_by_value(
          eta,
          etal_t,
          etau_t)
        p_t = p - m_t * eta_t

        self.updates.append(K.update(m, m_t))
        self.updates.append(K.update(v, v_t))
        self.updates.append(K.update(p, p_t))
      return self.updates