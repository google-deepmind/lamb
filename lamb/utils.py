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

"""Various utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import re
from absl import logging
import numpy as np
import six
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest


def ensure_list(x):
  return [x] if not isinstance(x, list) else x


def linear(args, output_size, bias, bias_start=0.0, initializer=None,
           scope=None):
  with tf.variable_scope(scope or 'linear', initializer=initializer):
    return _linear(args, output_size, bias, bias_start)


_BIAS_VARIABLE_NAME = 'bias'
_WEIGHTS_VARIABLE_NAME = 'kernel'


def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError('`args` must be specified')
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError('linear is expecting 2D arguments: %s' % shapes)
    if shape[1].value is None:
      raise ValueError('linear expects shape[1] to be provided for shape %s, '
                       'but saw %s' % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = tf.get_variable_scope()
  with tf.variable_scope(scope) as outer_scope:
    weights = tf.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return tf.add(res, biases)


def _find_var_init_param(var_init_params, var_name, key, default):
  if var_init_params is not None:
    for pattern in var_init_params:
      if re.match(pattern, var_name):
        value = var_init_params[pattern].get(key, None)
        if value is not None:
          return value
  return default


def find_initializer(var_init_params, name, default_initializer):
  return _find_var_init_param(var_init_params, name, 'initializer',
                              default_initializer)


def linear_v2(args, output_size, bias, rank=None, scope=None,   # pylint: disable=missing-docstring
              weight_name=_WEIGHTS_VARIABLE_NAME,
              bias_name=_BIAS_VARIABLE_NAME,
              var_init_params=None):
  with tf.variable_scope(scope or 'linear_v2'):
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError('`args` must be specified')
    if not nest.is_sequence(args):
      args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError('linear is expecting 2D arguments: %s' % shapes)
      if shape[1].value is None:
        raise ValueError('linear expects shape[1] to be provided for shape %s, '
                         'but saw %s' % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
      if rank is not None:
        if isinstance(rank, float):
          rank = int(min(total_arg_size, output_size)*rank)
        if rank == 0:
          rank = 1
        if rank >= min(total_arg_size, output_size):
          rank = None

      if rank is None:
        weights_initializer = find_initializer(
            var_init_params, weight_name, None)
        weights = tf.get_variable(
            weight_name, [total_arg_size, output_size], dtype=dtype,
            initializer=weights_initializer)
        if len(args) == 1:
          res = tf.matmul(args[0], weights)
        else:
          res = tf.matmul(tf.concat(args, 1), weights)
      else:
        left, right = low_rank_factorization(
            weight_name, [total_arg_size, output_size], rank)
        if len(args) == 1:
          res = tf.matmul(tf.matmul(args[0], left), right)
        else:
          res = tf.matmul(tf.matmul(tf.concat(args, 1), left), right)
      if not bias:
        return res
      with tf.variable_scope(outer_scope) as inner_scope:
        inner_scope.set_partitioner(None)
        bias_initializer = find_initializer(var_init_params, bias_name, None)
        biases = tf.get_variable(bias_name, [output_size], dtype=dtype,
                                 initializer=bias_initializer)
      return tf.add(res, biases)


def layer_norm(x, reduction_indices, epsilon=1e-9, gain=None, bias=None,
               per_element=True, scope=None):
  """DOC."""
  reduction_indices = ensure_list(reduction_indices)
  mean = tf.reduce_mean(x, reduction_indices, keep_dims=True)
  variance = tf.reduce_mean(tf.squared_difference(x, mean),
                            reduction_indices, keep_dims=True)
  normalized = (x - mean) / tf.sqrt(variance + epsilon)
  dtype = x.dtype
  shape = x.get_shape().as_list()
  for i in six.moves.range(len(shape)):
    if i not in reduction_indices or not per_element:
      shape[i] = 1
  with tf.variable_scope(scope or 'layer_norm'):
    if gain is None:
      gain = tf.get_variable('gain', shape=shape, dtype=dtype,
                             initializer=tf.ones_initializer())
    if bias is None:
      bias = tf.get_variable('bias', shape=shape, dtype=dtype,
                             initializer=tf.zeros_initializer())
  return gain*normalized+bias


def get_sparse_variable(name, indices, shape, dtype=None, trainable=True,
                        initializer=None, partitioner=None, regularizer=None):
  n = len(indices)
  values = tf.get_variable(name, [n], dtype=dtype,
                           initializer=initializer, partitioner=partitioner,
                           regularizer=regularizer, trainable=trainable)
  return tf.sparse_reorder(
      tf.SparseTensor(indices=indices, values=values, dense_shape=shape))


def _random_index(shape):
  if len(shape) == 0:  # pylint: disable=g-explicit-length-test
    return ()
  else:
    return (random.randrange(shape[0]),) + _random_index(shape[1:])


def _all_indices(shape):  # pylint: disable=missing-docstring
  indices = []
  n = len(shape)
  def add_indices(shape, depth, index):
    if depth == n:
      indices.append(index)
    else:
      for i in six.moves.range(shape[depth]):
        add_indices(shape, depth+1, index + [i])
  add_indices(shape, 0, [])
  return indices


def sparse_random_indices(ratio, shape):
  """DOC."""
  assert 0 < ratio and ratio <= 1.0
  n = round_to_int(tf.TensorShape(shape).num_elements()*ratio)
  # There are two implementations. The first generates random indices
  # and wastes computation due to collisions, and the second wastes
  # memory.
  if ratio < 0.25:
    indices = {}
    if isinstance(shape, tf.TensorShape):
      shape = shape.as_list()
    while len(indices) < n:
      index = _random_index(shape)
      indices[index] = True
    return indices.keys()
  else:
    indices = _all_indices(shape)
    random.shuffle(indices)
    return indices[:n]


def get_optimizer(optimizer_type):
  """Returns an optimizer builder.

  Args:
    optimizer_type: Short-form name of optimizer. Must be one of adadelta,
      adagrad, adam, rmsprop, sgd.

  Returns:
    A function of two arguments:
    - learning rate (tensor-like)
    - a tf.HParams object from which hyperparamters for optimization
      are extracted. Adam for example, extracts values of
      `adam_beta1`, `adam_beta2` and `adam_epsilon` from HParams.
      HParams doesn't need to have all listed keys.
  """
  fixed_args = {}
  keys = []
  prefix = optimizer_type
  if optimizer_type == 'adadelta':
    opt_func = tf.train.AdadeltaOptimizer
  elif optimizer_type == 'adagrad':
    opt_func = tf.train.AdagradOptimizer
  elif optimizer_type == 'adam':
    opt_func = tf.train.AdamOptimizer
    keys = ('beta1', 'beta2', 'epsilon')
  elif optimizer_type == 'rmsprop':
    opt_func = tf.train.AdamOptimizer
    fixed_args = {'beta1': 0.0}
    keys = ('beta2', 'epsilon')
  elif optimizer_type == 'sgd':
    opt_func = tf.train.GradientDescentOptimizer
  else:
    assert False

  logging.info('%s optimisation.', prefix)

  def build(learning_rate, config):
    args = _extract_dict_from_config(config, prefix + '_', keys)
    logging.info('%s hyperparameters: %r', prefix, args)
    return opt_func(learning_rate, **dict(args, **fixed_args))

  return build


def _extract_dict_from_config(config, prefix, keys):
  """Return a subset of key/value pairs from `config` as a dict.

  Args:
    config: A Config object.
    prefix: A string to which `keys` are added to form keys in `config`.
    keys: The potential keys in the resulting dict.

  Returns:
    A dict with `key`/`value` pairs where `prefix + key` has value
    `value` in `config`.
  """
  subset = {}
  for key in keys:
    config_key = prefix + key
    subset[key] = config[config_key]
  return subset


def repeat(iterable, n):
  """Repeats each element in iterable n times.

  Args:
    iterable: an iterable with elements to repeat.
    n: number of repetitions.

  Yields:
    Elements from `iterable` each repeated `n` times.
  """
  for e in iterable:
    for _ in range(n):
      yield e


def is_var_in_scope(var, scopes):
  for scope in ensure_list(scopes):
    # pylint: disable=g-explicit-bool-comparison
    if var.name.startswith(scope + '/') or var.name == scope or scope == '':
      return True
  return False


def trainable_vars_in_scope(name):
  """Return a list of variables in scope `name`.

  Args:
    name: The name of the scope without the trailing '/'.

  Returns:
    A list of tf.Variables in the scope.
  """
  vars_ = []
  for var in tf.trainable_variables():
    if is_var_in_scope(var, name):
      vars_.append(var)
  return vars_


def find_var(name, vars_=None):
  """Find a variable by name or return None.

  Args:
    name: The name of the variable (full qualified with all
      enclosing scopes).
    vars_: The variables among which to search. Defaults to all
      trainable variables.

  Returns:
    The [first] variable with `name` among `vars_` or None if there
    is no match.
  """
  if vars_ is None:
    vars_ = tf.trainable_variables()
  return next((var for var in vars_ if var.name == name),
              None)


def group_vars_by_scope(scopes, vars_=None, log=False):
  """Return a scope to list of vars in that scope map as a dict.

  Args:
    scopes: A sequence of scope names without the trailing '/'.
    vars_: The variables among which to search. Defaults to all
      trainable variables.
    log: Whether to log matching and ummatching variables
  Returns:

    A dictionary that maps a scope to variables among `vars_` in that
    scope. As the second value return all variables that were in none
    of `scopes`.
  """
  if vars_ is None:
    vars_ = tf.trainable_variables()
  var_groups = {}
  unmatched_vars = []
  for var in vars_:
    for scope in scopes:
      if is_var_in_scope(var, scope):
        if scope not in var_groups:
          var_groups[scope] = []
        var_groups[scope].append(var)
        if log:
          logging.info('%s -- %s', scope, var.name[len(scope):])
        break
    else:
      logging.info('-- %s', var.name)
      unmatched_vars.append(var)
  return var_groups, unmatched_vars


def _order_grouped_vars(var_groups):
  # pylint: disable=g-complex-comprehension
  return [var for scope in sorted(var_groups.keys())
          for var in var_groups[scope]]


def gradients_for_var_group(var_groups, gradients, name):
  """Returns a slice of `gradients` belonging to the var group `name`."""
  start = 0
  for group_name in sorted(var_groups.keys()):
    n = len(var_groups[group_name])
    if group_name == name:
      return gradients[start:start+n]
    start += n
  return []


def trainable_initial_state(batch_size, state_size, initial_state_init=None):
  """Make trainable initial state for an RNN cell with `state_size`."""
  def create_one(i, size):
    if initial_state_init is not None:
      initializer = initial_state_init
    else:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=1)
    return get_batched_variable(
        'initial_state_t{}'.format(i), batch_size, size,
        initializer=initializer)
  flat_vars = [create_one(i, size)
               for i, size in enumerate(nest.flatten(state_size))]
  return nest.pack_sequence_as(state_size, flat_vars)


def map_nested(fn, x):
  return nest.pack_sequence_as(x, list(map(fn, nest.flatten(x))))


def create_grads(optimizer, loss, scopes, num_expected_missing_gradients=0):
  """Compute, apply gradients and add summaries for norms."""
  logging.info('Creating gradient updates for scopes %r', scopes)
  grouped_vars, _ = group_vars_by_scope(scopes, log=True)
  ordered_vars = _order_grouped_vars(grouped_vars)
  grads_and_vars = optimizer.compute_gradients(
      loss, ordered_vars,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
  grads, _ = zip(*grads_and_vars)
  num_missing_grads = sum(grad is None for grad in grads)
  # Check that the gradient flow is not broken inadvertently. All
  # trainable variables should have gradients.
  if num_missing_grads > 0:
    for grad, var in grads_and_vars:
      if grad is None:
        logging.info('NO GRADIENT for var %s', var.name)
      else:
        logging.info('Gradients found for %s', var.name)
  assert num_missing_grads <= num_expected_missing_gradients, (
      '%s variables have no gradients. Expected at most %s.' %
      (num_missing_grads, num_expected_missing_gradients))
  summaries = []
  for grad, var in grads_and_vars:
    summaries.append(
        tf.summary.scalar(escape_summary_name(var.name + '_grad_norm'),
                          tf.norm(grad)))
  return grads_and_vars, summaries


def clip_gradients_in_scope(grads_and_vars, scope, max_grad_norm):
  """DOC."""
  if max_grad_norm == 0:
    return grads_and_vars
  else:
    grads_in_scope = []
    vars_in_scope = []
    for grad, var in grads_and_vars:
      if is_var_in_scope(var, scope):
        grads_in_scope.append(grad)
        vars_in_scope.append(var)
    clipped_grads_in_scope, _ = tf.clip_by_global_norm(
        grads_in_scope, max_grad_norm)
    new_grads_and_vars = []
    for grad, var in grads_and_vars:
      if vars_in_scope and var is vars_in_scope[0]:
        new_grads_and_vars.append((clipped_grads_in_scope.pop(0),
                                   vars_in_scope.pop(0)))
      else:
        new_grads_and_vars.append((grad, var))
    return new_grads_and_vars


def cv_splits(seq, k=10):
  """Split `seq` into `k` folds for cross-validation.

  Args:
    seq: A sequence of arbitrary elements.
    k: Positive integer. The number of folds.

  Yields:
    `k` (training, validation) elements. The validation component of
    each tuple is a subset of `seq` of length len(seq)/k (modulo
    rounding) while training has all remaining elements from `seq`.
    The validation sequences together form a segmentation of `seq`.
    Note that `seq` is shuffled before the splitting.

  """
  seq = list(seq)
  random.shuffle(seq)
  n = len(seq)
  m = float(n) / k
  start = 0
  for i in six.moves.range(k):
    if i == k -1:
      end = n
    else:
      end = int((i+1) * m)
    yield seq[:start] + seq[end:], seq[start:end]
    start = end


def escape_summary_name(name):
  return name.replace(':', '_')


def summaries_for_trainables():
  summaries = []
  for var in tf.trainable_variables():
    name = escape_summary_name(var.name)
    mean = tf.reduce_mean([var])
    summaries.append(tf.summary.scalar(name + '_mean', mean))
    summaries.append(tf.summary.scalar(name + '_var',
                                       tf.reduce_mean(tf.square([var-mean]))))
  return summaries


def log_scalar_summaries(summary_proto):
  summary = tf.Summary()
  summary.ParseFromString(summary_proto)
  for value in summary.value:
    logging.info('%s: %s', value.tag, value.simple_value)


def count_trainables(scopes=('',)):
  total = 0
  for scope in scopes:
    n = 0
    for var in trainable_vars_in_scope(scope):
      shape = var.get_shape()
      n += shape.num_elements()
    total += n
  return total


def log_trainables(scopes=('',)):
  """"Log number of trainable parameters for each scope in `scopes`.

  Args:
    scopes: A sequence of scope names.

  Returns:
    The total number of trainable parameters over all scopes in
    `scopes`. Possibly counting some parameters multiple times if the
    scopes are nested.
  """
  total = 0
  for scope in scopes:
    logging.info('Trainables in scope "%s":', scope)
    n = 0
    for var in trainable_vars_in_scope(scope):
      shape = var.get_shape()
      logging.info('trainable: %s shape %r (%r)', var.name, shape.as_list(),
                   shape.num_elements())
      n += shape.num_elements()
    logging.info('Number of parameters in scope "%s": %r', scope, n)
    total += n
  return total


def round_to_int(x):
  return int(round(x))


# tf.orthogonal_initializer is not stable numerically on some machines
# (gpus?).
def orthogonal_initializer(gain=1.0, dtype=tf.float32):
  """Generates orthonormal matrices with random values.

  Orthonormal initialization is important for RNNs:
    http://arxiv.org/abs/1312.6120
    http://smerity.com/articles/2016/orthogonal_init.html

  For non-square shapes the returned matrix will be semi-orthonormal: if the
  number of columns exceeds the number of rows, then the rows are orthonormal
  vectors; but if the number of rows exceeds the number of columns, then the
  columns are orthonormal vectors.

  We use SVD decomposition to generate an orthonormal matrix with random values.
  The same way as it is done in the Lasagne library for Theano. Note that both
  u and v returned by the svd are orthogonal and random. We just need to pick
  one with the right shape.

  Args:
    gain: A scalar with which the orthonormal tensor will be
      multiplied unless overridden at initialization time. The rule of
      thumb is 1.0 for weights before sigmoid activations, and sqrt(2)
      before RELUs, but tuning this might be required.
    dtype: a dtype of the intialized tensor.

  Returns:
    An initializer that generates random orthogonal matrices.
  """
  def initialize(shape, dtype=dtype, partition_info=None, gain=gain):
    partition_info = partition_info
    flat_shape = (shape[0], np.prod(shape[1:]))
    w = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(w, full_matrices=False)
    w = u if u.shape == flat_shape else v
    return tf.constant(gain*w.reshape(shape), dtype=dtype)
  return initialize


def random_mask(shape, k):
  x = tf.random_normal(shape=shape)
  kth_largest = tf.nn.top_k(x, k)[0][:, k-1]
  return tf.to_float(tf.greater_equal(x, tf.expand_dims(kth_largest, 1)))


def random_mask2(shape, k):
  x = tf.random_normal(shape=shape)
  x = tf.transpose(x)
  kth_largest = tf.nn.top_k(x, k)[0][:, k-1]
  mask = tf.to_float(tf.greater_equal(x, tf.expand_dims(kth_largest, 1)))
  return tf.transpose(mask)


def make_low_rank_factorization_initializer(shape, rank):
  fan_in = int(shape[0])
  # This is the variance we'd like to see if a matrix of 'shape' was
  # initialized directly.
  variance = 1.0 / fan_in
  # Each element of a*b (the low rank matrices) is the sum of 'rank'
  # terms, each of which is a product of an element from 'a' and
  # 'b'.
  stddev = np.sqrt(np.sqrt(variance / rank))
  return tf.initializers.truncated_normal(stddev=stddev)


def low_rank_factorization(name, shape, rank, initializer=None,
                           trainable=True, collections=None):
  # pylint: disable=missing-docstring
  if initializer is None:
    initializer = make_low_rank_factorization_initializer(shape, rank)
  a = tf.get_variable(
      name + '_a', [shape[0], rank],
      initializer=initializer, trainable=trainable, collections=collections)
  b = tf.get_variable(
      name + '_b', [rank, shape[1]],
      initializer=initializer, trainable=trainable, collections=collections)
  return a, b


class TFSerializer(object):
  """Serialize python object into a tf string variable."""

  def __init__(self, name='serialized', initializer='{}'):
    self._string = tf.get_variable(
        name, dtype=tf.string, trainable=False, initializer=initializer)
    self._new_string = tf.placeholder(dtype=tf.string, shape=[])
    self._assign_op = tf.assign(self._string, self._new_string)

  def store(self, obj):
    tf.get_default_session().run(self._assign_op,
                                 feed_dict={self._new_string: str(obj)})

  def retrieve(self):
    # pylint: disable=eval-used
    return eval(tf.get_default_session().run(self._string))

  def variables(self):
    return [self._string]


def mixture_of_softmaxes(x, k, e, to_logits):
  """A slower, but supposedly more flexible softmax.

  See "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model"
  by Yang et al, 2017.

  Args:
    x: A 2d tensor of shape [b, *]. Typically the output of an RNN cell.
    k: The number of mixture components.
    e: The embedding size. Often the same as the second dimension of x.
    to_logits: A function that takes a [b*k, e] tensor as its argument and
        transforms it into shape [b*k, v] where v is the vocabulary size.

  Returns:
    A [b, v] tensor of log probabilities. Each element is computed from
    the mixture of the k components. The components share most of the
    parameters (i.e. those in to_logits), but they have a smaller number
    of non-shared parameters (those in the projections).
  """
  # TODO(melisgl): For training where the entire output distribution is not
  # needed, maybe sparse_softmax_cross_entropy_with_logits would be more
  # efficient.
  if True:  # pylint: disable=using-constant-test
    # This log-domain implementation seems preferrable, but it uses much more
    # memory for some reason.
    b = tf.shape(x)[0]
    p_b_ke = tf.tanh(linear(x, k*e, True, scope='projection'))
    p_bk_e = tf.reshape(p_b_ke, [b*k, e])
    log_mixture_weights_b_k = tf.nn.log_softmax(
        linear(x, k, False, scope='mos_weights'))
    log_mixture_weights_b_k_1 = tf.reshape(log_mixture_weights_b_k, [b, k, 1])
    logits_bk_v = to_logits(p_bk_e)
    logprobs_bk_v = tf.nn.log_softmax(logits_bk_v)
    logprobs_b_k_v = tf.reshape(logprobs_bk_v, [b, k, -1])
    logprobs_b_v = tf.reduce_logsumexp(
        logprobs_b_k_v + log_mixture_weights_b_k_1,
        axis=1)
    return logprobs_b_v
  else:
    # Alternatively, calculate with probabilities directly.
    b = tf.shape(x)[0]
    p_b_ke = tf.tanh(linear(x, k*e, True, scope='projection'))
    p_bk_e = tf.reshape(p_b_ke, [b*k, e])
    mixture_weights_b_k = tf.nn.softmax(
        linear(x, k, False, scope='mos_weights'))
    mixture_weights_b_k_1 = tf.reshape(mixture_weights_b_k, [b, k, 1])
    logits_bk_v = to_logits(p_bk_e)
    probs_bk_v = tf.nn.softmax(logits_bk_v)
    probs_b_k_v = tf.reshape(probs_bk_v, [b, k, -1])
    probs_b_v = tf.reduce_sum(
        probs_b_k_v * mixture_weights_b_k_1,
        axis=1)
    return tf.log(probs_b_v+1e-8)


def expand_tile(tensor, n, name=None):
  """Returns a tensor repeated n times along a newly added first dimension."""
  with tf.name_scope(name, 'expand_tile'):
    n_ = tf.reshape(n, [1])
    num_dims = len(tensor.get_shape().as_list())
    multiples = tf.concat([n_, tf.ones([num_dims], dtype=tf.int32)], axis=0)
    # multiples = [n, 1, 1, ..., 1]
    res = tf.tile(tf.expand_dims(tensor, 0), multiples)

    first_dim = None
    if isinstance(n, int):
      first_dim = n
    res.set_shape([first_dim] + tensor.get_shape().as_list())

  return res


def get_batched_variable(name, runtime_batch_size, shape=None,
                         dtype=tf.float32, initializer=None,
                         trainable=True):
  """Returns a new variable tensor tiled runtime_batch_size number of times.

  Args:
    name: name for the new variable.
    runtime_batch_size: number of times to repeat the new variable along the
        first dimentsion.
    shape: shape of the new variable (e.g. [size] for [runtime_batch_size, size]
        output).
    dtype: type of the new variable to repeat.
    initializer: initializer for the variable.
    trainable: whether the new variable is trainable.

  Returns:
    A Tensor with variable of shape `shape` repeated `runtime_batch_size` times
    along added first dimension.
  """
  if initializer is None:
    initializer = tf.zeros_initializer(dtype=dtype)
  # If we're initializing from a constant then get_variable want a None shape.
  shape = None if isinstance(initializer, tf.Tensor) else shape
  var = tf.get_variable(name, shape=shape, dtype=dtype,
                        initializer=initializer, trainable=trainable)
  return expand_tile(var, runtime_batch_size)


# We should eventually merge mask_from_lengths and create_mask.
# mask_from_lengths is more robust since shapes only need to be known at runtime
# NB: create_mask is time major, whereas mask_from_lengths is batch major
def create_mask(lengths, max_length):
  """Created a mask of shape [time, batch_size] to mask out padding."""
  return tf.less(tf.reshape(tf.range(max_length, dtype=lengths.dtype),
                            [-1, 1]),
                 tf.reshape(lengths, [1, -1]))


def mask_from_lengths(lengths, max_length=None, dtype=None, name=None):
  """Convert a length scalar to a vector of binary masks.

  This function will convert a vector of lengths to a matrix of binary masks.
  E.g. [2, 4, 3] will become [[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]]

  Args:
    lengths: a d-dimensional vector of integers corresponding to lengths.
    max_length: an optional (default: None) scalar-like or 0-dimensional tensor
      indicating the maximum length of the masks. If not provided, the maximum
      length will be inferred from the lengths vector.
    dtype: the dtype of the returned mask, if specified. If None, the dtype of
      the lengths will be used.
    name: a name for the operation (optional).

  Returns:
    A d x max_length tensor of binary masks (int32).
  """
  with tf.name_scope(name, 'mask_from_lengths'):
    dtype = lengths.dtype if dtype is None else dtype
    max_length = tf.reduce_max(lengths) if max_length is None else max_length
    indexes = tf.range(max_length, dtype=lengths.dtype)
    mask = tf.less(tf.expand_dims(indexes, 0), tf.expand_dims(lengths, 1))
    cast_mask = tf.cast(mask, dtype)
  return tf.stop_gradient(cast_mask)


def compute_lengths(symbols_list, eos_symbol, name=None,
                    dtype=tf.int64):
  """Computes sequence lengths given end-of-sequence symbol.

  Args:
    symbols_list: list of [batch_size] tensors of symbols (e.g. integers).
    eos_symbol: end of sequence symbol (e.g. integer).
    name: name for the name scope of this op.
    dtype: type of symbols, default: tf.int64.

  Returns:
    Tensor [batch_size] of lengths of sequences.
  """
  with tf.name_scope(name, 'compute_lengths'):
    max_len = len(symbols_list)
    eos_symbol_ = tf.constant(eos_symbol, dtype=dtype)
    # Array with max_len-time where we have EOS, 0 otherwise. Maximum of this is
    # the first EOS in that example.
    ends = [tf.constant(max_len - i, dtype=tf.int64)
            * tf.to_int64(tf.equal(s, eos_symbol_))
            for i, s in enumerate(symbols_list)]
    # Lengths of sequences, or max_len for sequences that didn't have EOS.
    # Note: examples that don't have EOS will have max value of 0 and value of
    # max_len+1 in lens_.
    lens_ = max_len + 1 - tf.reduce_max(tf.stack(ends, 1), axis=1)
    # For examples that didn't have EOS decrease max_len+1 to max_len as the
    # length.
    lens = tf.subtract(lens_, tf.to_int64(tf.equal(lens_, max_len + 1)))
    return tf.stop_gradient(tf.reshape(lens, [-1]))


def seq_softmax_cross_entropy_with_logits(logits, labels, lengths,
                                          max_length=None, reduce_sum=True,
                                          name=None):
  """Softmax cross-entropy for a batch of sequences of varying lengths.

  The logits and labels arguments are similar to those of
  sparse_softmax_cross_entropy_with_logits with different shape
  requirements.

  Args:
    logits: Unscaled log probabilites of shape [time, batch, num_classes].
    labels: Indices of the true classes of shape [time, batch] and dtype
      int32 or int64.
    lengths: [batch], dtype int32 or int64
    max_length: Scalar integer. The time dimension in the above.
      Inferred if possible.
    reduce_sum: Whether to sum the cross entropy terms.
    name: name for the name scope of this op.

  Returns:
    The cross-entropy loss. If `reduce_sum`, then the shape is
    [batch], else it's the same shape as `labels`.
  """
  with tf.name_scope(name, 'seq_softmax_cross_entropy_with_logits',
                     [logits, labels, lengths, max_length]):
    mask = create_mask(lengths, max_length)
    # TODO(melisgl): Maybe call softmax_cross_entropy_with_logits
    # if the dtype of labels is non-integer.
    xe_terms = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    masked_xe_terms = xe_terms * tf.cast(mask, xe_terms.dtype)
    if reduce_sum:
      return tf.reduce_sum(masked_xe_terms, axis=0)
    else:
      return masked_xe_terms


def variance_scaling_initializer(scale=2.0, mode='fan_in',
                                 distribution='truncated_normal',
                                 mean=0.0, seed=None, dtype=tf.float32):
  """Like tf.variance_scaling_initializer but supports non-zero means."""
  if not dtype.is_floating:
    raise TypeError('Cannot create initializer for non-floating point type.')
  if mode not in ['fan_in', 'fan_out', 'fan_avg']:
    raise TypeError('Unknown mode %s [fan_in, fan_out, fan_avg]' % mode)

  # pylint: disable=unused-argument
  def _initializer(shape, dtype=dtype, partition_info=None):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'fan_in':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'fan_out':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'fan_avg':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0
    if distribution == 'truncated_normal':
      # To get stddev = math.sqrt(scale / n) need to adjust for truncated.
      trunc_stddev = math.sqrt(1.3 * scale / n)
      return tf.truncated_normal(shape, mean, trunc_stddev, dtype, seed=seed)
    elif distribution == 'uniform':
      # To get stddev = math.sqrt(scale / n) need to adjust for uniform.
      limit = math.sqrt(3.0 * scale / n)
      return tf.random_uniform(shape, mean-limit, mean+limit, dtype, seed=seed)
    else:
      assert 'Unexpected distribution %s.' % distribution
  # pylint: enable=unused-argument

  return _initializer


class FNCell(tf.nn.rnn_cell.RNNCell):
  """Dummy cell with no state that transforms its input with a function."""

  def __init__(self, fn, output_size, reuse=None):
    super(FNCell, self).__init__(_reuse=reuse)
    if output_size < 1:
      raise ValueError('Parameter output_size must be > 0: %d.' % output_size)
    self._fn = fn
    self._output_size = output_size

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
      return tf.zeros([batch_size, 1], dtype)

  def call(self, inputs, state):
    return self._fn(inputs), state
