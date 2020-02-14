# -*- coding: utf-8 -*-
"""
this .py file includes AMSGrad optimizer
"""

from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K

#----------------------------------------------------------

class AMSGrad(Optimizer):
  """Adam optimizer.

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
               epsilon=None, decay=0., amsgrad=True, **kwargs):
    super(AMSGrad, self).__init__(**kwargs)
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
    self.amsgrad = amsgrad

  def get_updates(self, loss, params):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.initial_decay > 0:
      lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                K.dtype(self.decay))))

    t = K.cast(self.iterations, K.floatx()) + 1
    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                 (1. - K.pow(self.beta_1, t)))
    beta_1_t = self.beta_1 / t

    ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    if self.amsgrad:
      vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    else:
      vhats = [K.zeros(1) for _ in params]
    self.weights = [self.iterations] + ms + vs + vhats

    for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
      m_t = (beta_1_t * m) + (1. - beta_1_t) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      if self.amsgrad:
        vhat_t = K.maximum(vhat, v_t)
        p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
        self.updates.append(K.update(vhat, vhat_t))
      else:
        p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))
      self.updates.append(K.update(p, p_t))
    return self.updates

