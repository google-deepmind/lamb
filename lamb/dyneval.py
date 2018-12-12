# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Dynamic evaluation."""

# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class Dyneval(object):

  def __init__(self, grads_and_vars, learning_rate, decay_rate, epsilon):
    with tf.variable_scope('dyneval'):
      # convert_to_tensor densifies IndexedSlices
      self._grads = [tf.convert_to_tensor(grad) for grad, _ in grads_and_vars]
      self._vars = [var for _, var in grads_and_vars]
      self._learning_rate = learning_rate
      self._decay_rate = decay_rate
      def shadow_vars():
        return [
            tf.get_variable(
                var.name.replace('/', '-').replace(':', '-'),
                var.get_shape(), initializer=tf.zeros_initializer(),
                trainable=False)
            for var in self._vars]
      with tf.variable_scope('save'):
        self._saves = shadow_vars()
      with tf.variable_scope('sum_squared_grads'):
        self._sum_squared_grads = shadow_vars()
      self._save = self._make_save()
      self._restore = self._make_restore()

      # These are for computing an RMSProplike estimate of the variance of
      # minibatch gradients. Here, this quantity is estimated on the training
      # set once, while gradient descent happens on validation/test.
      self._num_squared_grads = tf.get_variable(
          'num_squared_grads', [], initializer=tf.zeros_initializer(),
          trainable=False)
      self._zero_sum_squared_grads = self._make_zero_sum_squared_grads()
      self._add_squared_grads = self._make_add_squared_grads()
      self._epsilon = epsilon

      self._update = self._make_update()

  def _make_save(self):
    assignments = []
    for save, var in zip(self._saves, self._vars):
      assignments.append(save.assign(var))
    return tf.group(assignments)

  def _make_restore(self):
    assignments = []
    for save, var in zip(self._saves, self._vars):
      assignments.append(var.assign(save))
    return tf.group(assignments)

  def _make_update(self):
    mss = []
    gsum = 0.0
    count = 0
    for sum_squared_grads in self._sum_squared_grads:
      ms = tf.sqrt(sum_squared_grads / self._num_squared_grads)
      gsum += tf.reduce_sum(ms)
      count += tf.reduce_sum(tf.ones_like(ms))
      mss.append(ms)
    gsum = gsum / count

    assignments = []
    for grad, var, save, sum_squared_grads, ms in zip(
        self._grads, self._vars, self._saves, self._sum_squared_grads, mss):
      decay_rate = tf.minimum(1.0, self._decay_rate*(ms/gsum))
      delta = (-self._learning_rate*grad / (ms + self._epsilon) +
               decay_rate*(save-var))
      assignments.append(var.assign_add(delta))
    return tf.group(assignments)

  def _make_add_squared_grads(self):
    assignments = []
    for sum_squared_grads, grads in zip(self._sum_squared_grads, self._grads):
      assignments.append(sum_squared_grads.assign_add(tf.square(grads)))
    return tf.group(assignments + [self._num_squared_grads.assign_add(1)])

  def _make_zero_sum_squared_grads(self):
    assignments = []
    for sum_squared_grads in self._sum_squared_grads:
      assignments.append(sum_squared_grads.assign(
          tf.zeros_like(sum_squared_grads)))
    return tf.group(assignments + [self._num_squared_grads.assign(0)])

  def save(self):
    tf.get_default_session().run(self._save)

  def restore(self):
    tf.get_default_session().run(self._restore)

  def update_op(self):
    return self._update

  def zero_sum_squared_grads(self):
    tf.get_default_session().run(self._zero_sum_squared_grads)

  def add_squared_grads_op(self):
    return self._add_squared_grads

  def __enter__(self):
    self.save()

  def __exit__(self, type_, value, traceback):
    self.restore()
