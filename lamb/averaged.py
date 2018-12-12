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

"""Averaging of model weights."""

# pylint: disable=missing-docstring
# pylint: disable=g-complex-comprehension

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class Averaged(object):

  def __init__(self, tensors):
    tensors = list(tensors)
    with tf.variable_scope('averaged'):
      self._num_samples = tf.Variable(0, name='num_samples', trainable=False)
      with tf.variable_scope('avg'):
        self._averages = [
            tf.get_variable(
                tensor.name.replace('/', '-').replace(':', '-'),
                tensor.get_shape(), initializer=tf.zeros_initializer(),
                trainable=False)
            for tensor in tensors]
      with tf.variable_scope('save'):
        self._saves = [
            tf.get_variable(
                tensor.name.replace('/', '-').replace(':', '-'),
                tensor.get_shape(), initializer=tf.zeros_initializer(),
                trainable=False)
            for tensor in tensors]
    self._tensors = tensors
    self._take_sample = self._make_take_sample()
    self._switch = self._make_swith_to_average()
    self._restore = self._make_restore()
    self._reset = self._make_reset()

  def take_sample(self):
    tf.get_default_session().run(self._take_sample)

  def switch_to_average(self):
    tf.get_default_session().run(self._switch)

  def restore(self):
    tf.get_default_session().run(self._restore)

  def reset(self):
    tf.get_default_session().run(self._reset)

  def __enter__(self):
    self.switch_to_average()

  def __exit__(self, type_, value, traceback):
    self.restore()

  def _make_take_sample(self):
    assignments = []
    n = tf.cast(self._num_samples, tf.float32)
    mu = 1.0 / (1.0 + n)
    for tensor, average in zip(self._tensors, self._averages):
      assignments.append(tf.assign_add(average, (tensor-average)*mu))
    add_to_averages = tf.group(assignments)
    with tf.control_dependencies([add_to_averages]):
      incr_num_samples = tf.assign(self._num_samples, self._num_samples + 1)
    return incr_num_samples

  def _make_swith_to_average(self):
    assignments = []
    for save, tensor, average in zip(
        self._saves, self._tensors, self._averages):
      with tf.control_dependencies([save.assign(tensor)]):
        assignments.append(tensor.assign(average))
    return tf.group(assignments)

  def _make_restore(self):
    assignments = []
    for save, tensor in zip(self._saves, self._tensors):
      assignments.append(tf.assign(tensor, save))
    return tf.group(assignments)

  def _make_reset(self):
    return tf.assign(self._num_samples, 0)


# TODO(melisgl): I think this works with ResourceVariables but not with normal
# Variables. Deferred until TF2.0.
def _swap(x, y):
  x_value = x.read_value()
  y_value = y.read_value()
  with tf.control_dependencies([x_value, y_value]):
    swap = tf.group(y.assign(x_value), x.assign(y_value))
  return swap
