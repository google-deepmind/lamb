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

"""Variational Dropout."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.python.modules import base as snt_base
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow.contrib import util as contrib_util


class Dropout(snt_base.AbstractModule):
  """Possibly variational dropout."""

  def __init__(self, keep_prob, share_mask=True, scaler=1.0, name='dropout'):
    super(Dropout, self).__init__(name=name)
    self._keep_prob = keep_prob
    self._keep_mask = None
    self._share_mask = share_mask
    self._scaler = scaler

  def _ensure_keep_mask(self, x):
    if self._keep_mask is None or not self._share_mask:
      shape = tf.shape(x)
      noise = tf.random_uniform(shape, dtype=x.dtype)
      self._keep_mask = (tf.floor(self._keep_prob + noise)
                         * (self._scaler / self._keep_prob))
      self._keep_mask.set_shape(x.get_shape())
    return self._keep_mask

  def _build(self, x):
    if contrib_util.constant_value(self._keep_prob) == 1:
      return x
    else:
      return x * self._ensure_keep_mask(x)


class GaussianDropout(snt_base.AbstractModule):
  """Possibly variational dropout."""

  def __init__(self, keep_prob, share_mask=True, scaler=1.0, name='dropout'):
    super(GaussianDropout, self).__init__(name=name)
    self._keep_prob = keep_prob
    self._keep_mask = None
    self._share_mask = share_mask
    self._scaler = scaler

  def _ensure_keep_mask(self, x):
    if self._keep_mask is None or not self._share_mask:
      shape = tf.shape(x)
      # Calculate the stddev for the normal distribution that
      # matches the stddev of the bernoulli with p=keep_prob.
      stddev = tf.sqrt((1 - self._keep_prob) / self._keep_prob)
      self._keep_mask = tf.random_normal(shape, mean=1.0, stddev=stddev,
                                         dtype=x.dtype)
      self._keep_mask.set_shape(x.get_shape())
    return self._keep_mask

  def _build(self, x):
    if contrib_util.constant_value(self._keep_prob) == 1:
      return x
    else:
      return x * self._ensure_keep_mask(x)


class DirichletDropout(snt_base.AbstractModule):
  """Possibly variational dropout."""

  def __init__(self, keep_prob, share_mask=True, scaler=1.0, name='dropout'):
    super(DirichletDropout, self).__init__(name=name)
    self._keep_prob = keep_prob
    self._keep_mask = None
    self._share_mask = share_mask
    self._scaler = scaler

  def _ensure_keep_mask(self, x):
    if self._keep_mask is None or not self._share_mask:
      shape = tf.shape(x)
      k = shape[1]
      # To make this class a drop-in replacement for bernoulli dropout we
      # paramaterize it with keep_prob. Set alpha of the dirichlet so that the
      # variance is equal to the variance of the bernoulli with p=keep_prob
      # divided by keep_prob.
      # Now the variance of the dirichlet with k equal alphas is
      # (k-1)/(k^2(k*alpha+1). Solve that for alpha.
      kf = tf.cast(k, tf.float32)
      alpha = self._keep_prob * (kf - 1.0) / ((1-self._keep_prob)*kf) - 1.0/kf
      dist = tfp.distributions.Dirichlet(tf.ones(shape=k) * alpha)
      assert (dist.reparameterization_type ==
              tfp.distributions.FULLY_REPARAMETERIZED)
      # The E[dir(alpha)] = 1/k for all elements, but we want the expectation to
      # be keep_prob, hence the multiplication.
      self._keep_mask = kf * dist.sample(shape[0])
      self._keep_mask.set_shape(x.get_shape())
    return self._keep_mask

  def _build(self, x):
    if contrib_util.constant_value(self._keep_prob) == 1:
      return x
    else:
      return tf.cond(tf.equal(self._keep_prob, 1.0),
                     lambda: x,
                     lambda: x * self._ensure_keep_mask(x))


class DriftingDropout(snt_base.AbstractModule):
  """Dropout with gradually changing mask."""

  def __init__(self, keep_prob, flip_prob=0.0, scaler=1.0, name='dropout'):
    super(DriftingDropout, self).__init__(name=name)
    self._keep_prob = keep_prob
    self._flip_prob = flip_prob
    self._scaler = scaler
    self._time_step = 0

  def _build(self, x, state):
    prev_keep_mask = state
    shape = tf.shape(x)
    noise = tf.random_uniform(shape, dtype=x.dtype)
    other_mask = tf.floor(self._keep_prob + noise)
    choice_noise = tf.random_uniform(shape, dtype=x.dtype)
    choice = tf.less(choice_noise, self._flip_prob)
    # KLUDGE(melisgl): The client has to pass the last keep_mask from
    # a batch to the next so the mask may end up next to some
    # recurrent cell state. This state is often zero at the beginning
    # and may be periodically zeroed (per example) during training.
    # While zeroing LSTM state is okay, zeroing the dropout mask is
    # not. So instead of forcing every client to deal with this common
    # (?) case, if an all zero mask is detected, then regenerate a
    # fresh mask. This is of course a major hack and won't help with
    # learnt initial states, for example.
    sum_ = tf.reduce_sum(prev_keep_mask, 1, keepdims=True)
    is_initializing = tf.equal(sum_, 0.0)

    self._keep_mask = tf.where(tf.logical_or(choice, is_initializing),
                               other_mask,
                               prev_keep_mask)
    self._time_step += 1
    return x * self._keep_mask / self._keep_prob * self._scaler
