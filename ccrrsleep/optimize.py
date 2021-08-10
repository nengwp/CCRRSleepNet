'''
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
'''


import tensorflow as tf

"""AMSGrad for TensorFlow."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer


class AMSGrad(optimizer.Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        var = var.handle
        beta1_power = math_ops.cast(self._beta1_power, grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m").handle
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v").handle
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat").handle
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)
        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

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


def adam(loss, lr, train_vars, beta1=0.9, beta2=0.999, epsilon=1e-8):
    opt = AMSGrad(
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    grads_and_vars = opt.compute_gradients(loss, train_vars)
    apply_gradient_op = opt.apply_gradients(grads_and_vars)
    return apply_gradient_op, grads_and_vars


def adam_clipping(loss, lr, train_vars, beta1=0.9, beta2=0.999,
                  epsilon=1e-8, clip_value=5.0):
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars),
                                      clip_value)
    capped_gvs = list(zip(grads, train_vars))
    opt = AMSGrad(
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    apply_gradient_op = opt.apply_gradients(capped_gvs)
    return apply_gradient_op, capped_gvs


def adam_clipping_list_lr(loss, list_lrs, list_train_vars,
                          beta1=0.9, beta2=0.999,
                          epsilon=1e-8, clip_value=5.0):
    assert len(list_lrs) == len(list_train_vars)

    train_vars = []
    for v in list_train_vars:
        if len(train_vars) == 0:
            train_vars = list(v)
        else:
            train_vars.extend(v)

    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars),
                                      clip_value)

    offset = 0
    apply_gradient_ops = []
    grads_and_vars = []
    for i, v in enumerate(list_train_vars):
        g = grads[offset:offset+len(v)]
        opt = AMSGrad(
            learning_rate=list_lrs[i],
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        gvs = list(zip(g, v))
        apply_gradient_op = opt.apply_gradients(gvs)

        apply_gradient_ops.append(apply_gradient_op)
        if len(grads_and_vars) == 0:
            grads_and_vars = list(gvs)
        else:
            grads_and_vars.extend(gvs)
        offset += len(v)

    apply_gradient_ops = tf.group(*apply_gradient_ops)
    return apply_gradient_ops, grads_and_vars


#
#
# def adam(loss, lr, train_vars, beta1=0.9, beta2=0.999, epsilon=1e-8):
#     opt = tf.compat.v1.train.AdamOptimizer(
#         learning_rate=lr,
#         beta1=beta1,
#         beta2=beta2,
#         epsilon=epsilon,
#         name="Adam"
#     )
#     grads_and_vars = opt.compute_gradients(loss, train_vars)
#     apply_gradient_op = opt.apply_gradients(grads_and_vars)
#     return apply_gradient_op, grads_and_vars
#
#
# def adam_clipping(loss, lr, train_vars, beta1=0.9, beta2=0.999,
#                   epsilon=1e-8, clip_value=5.0):
#     grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars),
#                                       clip_value)
#     capped_gvs = list(zip(grads, train_vars))
#     opt = tf.compat.v1.train.AdamOptimizer(
#         learning_rate=lr,
#         beta1=beta1,
#         beta2=beta2,
#         epsilon=epsilon,
#         name="Adam"
#     )
#     apply_gradient_op = opt.apply_gradients(capped_gvs)
#     return apply_gradient_op, capped_gvs
#
#
# def adam_clipping_list_lr(loss, list_lrs, list_train_vars,
#                           beta1=0.9, beta2=0.999,
#                           epsilon=1e-8, clip_value=5.0):
#     assert len(list_lrs) == len(list_train_vars)
#
#     train_vars = []
#     for v in list_train_vars:
#         if len(train_vars) == 0:
#             train_vars = list(v)
#         else:
#             train_vars.extend(v)
#
#     grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars),
#                                       clip_value)
#
#     offset = 0
#     apply_gradient_ops = []
#     grads_and_vars = []
#     for i, v in enumerate(list_train_vars):
#         g = grads[offset:offset+len(v)]
#         opt = tf.compat.v1.train.AdamOptimizer(
#             learning_rate=list_lrs[i],
#             beta1=beta1,
#             beta2=beta2,
#             epsilon=epsilon,
#             name="Adam"
#         )
#         gvs = list(zip(g, v))
#         apply_gradient_op = opt.apply_gradients(gvs)
#
#         apply_gradient_ops.append(apply_gradient_op)
#         if len(grads_and_vars) == 0:
#             grads_and_vars = list(gvs)
#         else:
#             grads_and_vars.extend(gvs)
#         offset += len(v)
#
#     apply_gradient_ops = tf.group(*apply_gradient_ops)
#     return apply_gradient_ops, grads_and_vars