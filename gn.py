from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.training.gradient_descent import *


class GNOptimizer(GradientDescentOptimizer):
  def __init__(self, loss_tensor, pred_tensor, learning_rate=0.001, beta1=0.9, 
               beta2=0.999, epsilon=1e-8, use_locking=False, name="GN"):
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor versions of the constructor arguments, created in _prepare().
    # self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Tensors needed to calculate the update step
    self._loss_t = loss_tensor
    self._pred_t = pred_tensor


    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None
    super(GNOptimizer, self).__init__(learning_rate, use_locking, name)


  def _apply_sparse(self, grad, var):
    beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    


    lr = (lr_t * (1 - beta2_power) / (1 - beta1_power))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad.values * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    m_t = state_ops.scatter_add(m_t, grad.indices, m_scaled_g_values,
                               use_locking=self._use_locking)
    # gn_c = ((d/dy) * dLdy) * dydvar ** 2
    # gn_t = beta2 * gn + (1 - beta2) * (gn_c)
    dLdy = tf.gradients(self._loss_t, self._pred_t)
    sec_loss = tf.gradients(dLdy, self._pred_t)
    sec_loss_t = math_ops.cast(self, sec_loss, var.dtype.base_dtype)
    dydvar = tf.gradients(self.pred_t, var)
    dydvar_t = math_ops.cast(self, dydvar, var.dtype.base_dtype)
    gn_c = sec_loss_t * dydvar_t * dydvar_t
    gn = self.get_slot(var, "gn")
    gn_scaled_g_values = (gn_c) * (1 - beta2_t)
    gn_t = state_ops.assign(gn, gn * beta2_t, use_locking=self._use_locking)
    gn_t = state_ops.scatter_add(gn_t, grad.indices, gn_scaled_g_values,
                               use_locking=self._use_locking)
    var_update = state_ops.assign_sub(var,
                                      lr * m_t / (gn_t + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])


  def _apply_dense(self, grad, var):
    """
    Construct a new GN optimizer.

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    gn_0 <- 0 (Initialize initial 2nd moment vector)
    t <- 0 (Initialize timestep)
    ```

    This update rule will look very similiar to Adam, but uses the 2nd
    derivative of the Loss and does not use the sqrt

    ```
    t <- t + 1
    lr_t <- learning_rate * (1 - beta2^t) / (1 - beta1^t)
    
    dLdy <- tf.gradients(loss, y_hat)
    (d/dy) * dLdy <- tf.gradients(dLdy, y_hat)
    dydvar <- tf.gradients(y_hat, var)
    
    gn_curr <- ((d/dy) * dLdy) * dydvar ** 2


    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    gn_t <- beta2 * v_{t-1} + (1 - beta2) * gn_curr
    variable <- variable - lr_t * m_t / (gn_t + epsilon)
    ```

    The default value of 1e-8 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1.

    Note that in dense implement of this algorithm, m_t, gn_t and variable will 
    update even if g is zero, but in sparse implement, m_t, v_t and variable 
    will not update in iterations g is zero.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "GN".
    """

    beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
    lr_t = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    grad = math_ops.cast(grad, var.dtype.base_dtype)
    print(type(grad), grad, "grad")

    lr = (lr_t * (1 - beta2_power) / (1 - beta1_power))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    m_t = state_ops.assign_add(m_t, m_t + m_scaled_g_values,
                               use_locking=self._use_locking)
    # gn_c = ((d/dy) * dLdy) * dydvar ** 2
    # gn_t = beta2 * gn + (1 - beta2) * (gn_c)
    gn = tf.zeros(var.get_shape())
    y_flatten = tf.reshape(self._pred_t, [-1])
    for y 
    print(y_flatten, type(y_flatten))
    sec_dir = lambda x: tf.gradients(tf.gradients([x], x), x)[0]
    diag_hessian = lambda x: tf.gradients(x, var)[0] ** 2
    gn_c = tf.foldl(lambda a, x: a +diag_hessian(x), y_flatten)

    tf.while_loop(lambda)
    print(gn_c.get_shape())
    # dLdy = tf.gradients([self._loss_t], self._pred_t)
    # print(self._pred_t.get_shape(), len(dLdy), dLdy[0].get_shape())
    
    # # print(dLdy.get_shape(), self._loss_t.get_shape())
    # sec_loss = []
    # for i in range(len(dLdy)):
    #   sec_loss.append(tf.gradients([dLdy[i]], self._pred_t[i])[0])
    # print(len(sec_loss))
    # sec_loss_t = tf.pack(sec_loss, axis = 0)
    # # print("sec_loss: ", type(sec_loss), sec_loss.get_shape())
    # sec_loss_t = math_ops.cast(self, sec_loss_t, var.dtype.base_dtype)
    # dydvar = tf.gradients(self._pred_t, var)[0]
    # # print(dydvar, type(dydvar), type(dydvar[0]), var.dtype.base_dtype)
    # # print(dydvar[0].get_shape(), var.get_shape(), self._pred_t.get_shape(), len(dydvar))
    # dydvar_t = math_ops.cast(self, dydvar, var.dtype.base_dtype)
    # gn_c = sec_loss_t * dydvar_t * dydvar_t
    gn = self.get_slot(var, "gn")
    gn_scaled_g_values = (gn_c) * (1 - beta2_t)
    gn_t = state_ops.assign(gn, gn * beta2_t, use_locking=self._use_locking)
    gn_t = state_ops.assign_add(gn_t, gn_t + gn_scaled_g_values,
                               use_locking=self._use_locking)
    var_update = state_ops.assign_sub(var,
                                      lr * m_t / (gn_t + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])


    # new_grad = 
    # return training_ops.apply_gradient_descent(
    #     var,
    #     math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
    #     new_grad,
    #     use_locking=self._use_locking).op
  def _prepare(self):
    self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    super(GNOptimizer, self)._prepare()

  def _get_beta_accumulators(self):
    return self._beta1_power, self._beta2_power

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable.
    if (self._beta1_power is None or
        self._beta1_power.graph is not var_list[0].graph):
      with ops.colocate_with(var_list[0]):
        self._beta1_power = variables.Variable(self._beta1,
                                               name="beta1_power",
                                               trainable=False)
        self._beta2_power = variables.Variable(self._beta2,
                                               name="beta2_power",
                                               trainable=False)
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "gn", self._name)
  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._beta1_power):
        update_beta1 = self._beta1_power.assign(
            self._beta1_power * self._beta1_t,
            use_locking=self._use_locking)
        update_beta2 = self._beta2_power.assign(
            self._beta2_power * self._beta2_t,
            use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                  name=name_scope)
