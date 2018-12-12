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

"""Efficient linear mappings from a number of inputs to outputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl import logging
from lamb import utils
import six
from sonnet.python.modules import base as snt_base
import tensorflow.compat.v1 as tf


class AbstractTiledLinear(snt_base.AbstractModule):
  """Efficient linear mappings from a number of inputs to outputs."""

  def __init__(self, input_name_and_sizes, output_name_and_sizes,
               var_init_params=None, name='tiled_linear'):
    """Constructs a AbstractTiledLinear module.

    Args:
      input_name_and_sizes: A sequence of `(name, size)` tuples
        listing the inputs are their sizes (a positive integer or None
        to rely on shape inferencing at build() time). As a
        convenience, `(name, None)` can be shortened to `(name,)` or
        just `name`.
      output_name_and_sizes: Similar to `input_name_and_sizes`, it
        lists the names and sizes of outputs. Since there is no way of
        inferring shapes for outputs, the full `(name, size)` form
        must always be used.
      var_init_params: A dict for specifying initialization parameters
        for variables such as the initializer, partitioner and
        regularizer. Subclasses may support more parameters.
      name: Name of the module.

    Raises:
      ValueError: If ` contains any keys other than 'w' or 'b'.
      KeyError: If `partitioners` contains any keys other than 'w' or 'b'.
      KeyError: If `regularizers` contains any keys other than 'w' or 'b'.
      TypeError: If any of the given initializers are not callable.
      TypeError: If any of the given partitioners are not callable.
      TypeError: If any of the given regularizers are not callable.
    """
    super(AbstractTiledLinear, self).__init__(name=name)
    self._input_name_and_sizes = self._canonicalize_input_name_and_sizes(
        input_name_and_sizes)
    self._output_name_and_sizes_ = self._check_output_name_and_sizes(
        output_name_and_sizes)
    self._var_init_params = self._check_var_init_params(var_init_params)
    self._merged_input_sizes = None
    self._name = name
    self._dtype = None

  def _canonicalize_input_name_and_sizes(self, name_and_sizes):
    result = []
    for e in name_and_sizes:
      if isinstance(e, six.string_types):
        result.append((e, None))
      else:
        assert isinstance(e, tuple)
        if len(e) == 1:
          result.append((e[0], None))
        elif len(e) == 2:
          result.append(e)
        else:
          assert False, 'Malformed name_and_sizes spec {}.'.format(e)
    return result

  def _check_output_name_and_sizes(self, name_and_sizes):
    for e in name_and_sizes:
      assert isinstance(e, tuple)
      assert len(e) == 2
      assert isinstance(e[0], six.string_types)
      assert isinstance(e[1], int)
    return name_and_sizes

  def _check_var_init_params(self, var_init_params):
    if var_init_params is None:
      return {}
    else:
      valid_keys = self.valid_var_init_param_keys()
      for pattern in var_init_params:
        for key in var_init_params[pattern]:
          assert key in valid_keys, (
              'Unexpected key {} in var_init_params[{}].'.format(key, pattern))
      return var_init_params

  def _check_dtype(self, inputs, previous_dtype):
    dtype = previous_dtype
    for input_ in inputs:
      if dtype is None:
        dtype = input_.dtype
      else:
        assert input_.dtype == dtype
    return dtype

  def valid_var_init_param_keys(self):
    return ['initializer', 'partitioner', 'regularizer']

  def _find_var_init_param(self, var_name, key, default):
    for pattern in self._var_init_params:
      if re.match(pattern, var_name):
        value = self._var_init_params[pattern].get(key, None)
        if value is not None:
          return value
    return default

  def _get_variable(self, name, shape,
                    initializer=None,
                    default_initializer=None, default_partitioner=None,
                    default_regularizer=None):
    if initializer is None:
      initializer = self._find_var_init_param(
          name, 'initializer', default_initializer)
    partitioner = self._find_var_init_param(
        name, 'partitioner', default_partitioner)
    regularizer = self._find_var_init_param(
        name, 'regularizer', default_regularizer)
    return tf.get_variable(name, shape=shape, dtype=self._dtype,
                           initializer=initializer, partitioner=partitioner,
                           regularizer=regularizer)

  def _declared_input_sizes(self):
    sizes = []
    for _, input_size in self._input_name_and_sizes:
      sizes.append(input_size)
    return tf.TensorShape(sizes)

  def _inferred_input_sizes(self, inputs):
    return tf.TensorShape([input_.get_shape().as_list()[-1]
                           for input_ in inputs])

  def _merge_input_sizes(self, inputs):
    inferred_input_sizes = self._inferred_input_sizes(inputs)
    if self._merged_input_sizes is None:
      declared_input_sizes = self._declared_input_sizes()
      # This is the first call to build(). Remember the input sizes
      # (only the last dimension matters for matmul).
      if not declared_input_sizes.is_compatible_with(inferred_input_sizes):
        raise snt_base.IncompatibleShapeError(
            '{}: Declared input sizes {} are incompatible '
            'with inferred ones {}.'.format(
                self.scope_name, declared_input_sizes.as_list(),
                inferred_input_sizes.as_list()))
      self._merged_input_sizes = declared_input_sizes.merge_with(
          inferred_input_sizes)
      if not self._merged_input_sizes.is_fully_defined():
        raise snt_base.IncompatibleShapeError(
            '{}: Last input dimensions must be known at module build time.'
            ' Got {}.'.format(self.name, self._merged_input_sizes.as_list()))
    else:
      # At subsequent calls check that input sizes are compatible.
      if not self._merged_input_sizes.is_compatible_with(inferred_input_sizes):
        raise snt_base.IncompatibleShapeError(
            '{}: Current input sizes {} are different '
            'from first build {}'.format(
                self.name, inferred_input_sizes.as_list(),
                self._merged_input_sizes.as_list()))

  def _merged_input_name_and_sizes(self):
    return zip([input_name for input_name, _ in self._input_name_and_sizes],
               self._merged_input_sizes.as_list())

  def _output_name_and_sizes(self):
    return self._output_name_and_sizes_

  def _build(self, inputs):
    """Connects the module into the graph, with `inputs`.

    If this is not the first time the module has been connected to the
    graph, the Tensors in `inputs` must have the same final dimension,
    in order for the existing variables to be the correct size for the
    multiplication. The leading dimensions of the input tensors may
    differ for each call to `build()`.

    Args:
      inputs: A sequence of tensors. The last dimension of the tensor
        at position I must be compatible with the declared size of the
        corresponding input (if not None) and also with the last
        dimension of the corresponding input tensor in all previous
        calls to build() on the same object.

    Returns:
      A sequence of output tensors.

    Raises:
      base.IncompatibleShapeError: If the input sizes are not
        compatible with the declared or with the sizes previous calls.
    """
    self._merge_input_sizes(inputs)
    self._dtype = self._check_dtype(inputs, self._dtype)
    return self._build_tiled_linear(inputs,
                                    self._merged_input_name_and_sizes(),
                                    self._output_name_and_sizes(),
                                    True)


class TiledLinear(AbstractTiledLinear):
  """Plain linear mapping without any bells or whistles."""

  def __init__(self, input_name_and_sizes, output_name_and_sizes,
               var_init_params=None, name='tiled_linear'):
    """Plain linear mapping without any bells or whistles."""
    super(TiledLinear, self).__init__(
        input_name_and_sizes, output_name_and_sizes,
        var_init_params=var_init_params, name=name)
    self._weights = None
    self._biases = None

  def _ensure_weights(self):
    # pylint: disable=missing-docstring
    if self._weights is None:
      # Tile an initializer together from the initializers of the individual
      # tiles. We used to assemble the weight matrix by tiling the individual
      # matrices, but with that tensorflow wasted gobs of memory for the
      # gradients.
      default_initializer = utils.variance_scaling_initializer(scale=1.0)
      columns = []
      for output_name, output_size in self._output_name_and_sizes_:
        # Collect the initializers for the tiles for weight matrices mapping
        # _to_ the output being considered. These will be stacked in a column of
        # the final tiled weight matrix.
        initializers_to_output = []
        for input_name, input_size in self._input_name_and_sizes:
          name = 'W_{}_{}'.format(input_name, output_name)
          initializer = self._find_var_init_param(
              name, 'initializer', default_initializer)
          shape = [int(input_size), int(output_size)]
          # logging.info('Tile initializer for %r %r: %r',
          #              name, shape, initializer)
          initializers_to_output.append((initializer, shape))
        columns.append(initializers_to_output)
      def tiled_initializer(shape, dtype=self._dtype, partition_info=None):
        column_values = []
        for column in columns:
          values = [initializer(shape, dtype=dtype,
                                partition_info=partition_info)
                    for initializer, shape in column]
          column_values.append(tf.concat(values, axis=0))
        return tf.concat(column_values, axis=1)
      # Finally, instantiate the weights.
      total_input_size = sum([input_size for _, input_size
                              in self._input_name_and_sizes])
      total_output_size = sum([output_size for _, output_size
                               in self._output_name_and_sizes_])
      self._weights = self._get_variable(
          'W', shape=[total_input_size, total_output_size],
          initializer=tiled_initializer)
    return self._weights

  def _ensure_biases(self):
    # pylint: disable=missing-docstring
    if self._biases is None:
      # Biases are much smaller than weights, so wasting memory with gradients
      # is not an issue.
      biases = []
      for output_name, output_size in self._output_name_and_sizes_:
        bias = self._get_variable(
            'B_{}'.format(output_name), shape=[output_size],
            default_initializer=tf.zeros_initializer())
        biases.append(bias)
      self._biases = tf.concat(biases, 0)
    return self._biases

  def _build_tiled_linear(self, inputs, input_name_and_sizes,
                          output_name_and_sizes, add_bias):
    # pylint: disable=missing-docstring
    def split_output(output):
      if len(output_name_and_sizes) == 1:
        return output
      elif len(set([size for _, size in output_name_and_sizes])) == 1:
        # This is a bit faster than several tf.slice calls.
        return tf.split(output, len(output_name_and_sizes), axis=1)
      else:
        outputs = []
        offset = 0
        for _, output_size in output_name_and_sizes:
          outputs.append(tf.slice(output, [0, offset], [-1, output_size]))
          offset += output_size
        return outputs

    weights = self._ensure_weights()
    if len(inputs) > 1:
      inputs = tf.concat(inputs, 1)
    if add_bias:
      biases = self._ensure_biases()
      return split_output(tf.nn.xw_plus_b(inputs, weights, biases))
    else:
      return split_output(tf.matmul(inputs, weights))


class LayerNormedTiledLinear(AbstractTiledLinear):
  # pylint: disable=missing-docstring

  def _build_tiled_linear(self, inputs, input_name_and_sizes,
                          output_name_and_sizes, add_bias):
    # pylint: disable=missing-docstring

    # Return a list of weight matrices that parallels
    # input_name_and_sizes and maps one input tensor to the
    # concatenation of all outputs.
    def make_weights_for_inputs():
      rows = []
      for input_name, input_size in input_name_and_sizes:
        # Collect the weight matrices mapping from the input being
        # considered. These will be stacked in a row.
        weights_from_input = []
        for output_name, output_size in output_name_and_sizes:
          name = 'W_{}_{}'.format(input_name, output_name)
          weight = self._get_variable(name, shape=[input_size, output_size])
          weights_from_input.append(weight)
        rows.append(tf.concat(weights_from_input, 1))
      return rows

    def make_biases():
      biases = []
      for name, size in output_name_and_sizes:
        bias = self._get_variable('B_{}'.format(name), shape=[size],
                                  default_initializer=tf.zeros_initializer())
        biases.append(bias)
      return tf.concat(biases, 0)

    def split_output(output):
      outputs = []
      offset = 0
      for _, output_size in output_name_and_sizes:
        outputs.append(tf.slice(output, [0, offset], [-1, output_size]))
        offset += output_size
      return outputs

    weights_for_inputs = make_weights_for_inputs()

    s = make_biases() if add_bias else 0.0
    for input_, weights, (name, _) in zip(inputs, weights_for_inputs,
                                          input_name_and_sizes):
      s += utils.layer_norm(tf.matmul(input_, weights), [1], bias=0.0,
                            scope='ln_{}'.format(name))

    return split_output(s)


class SparseTiledLinear(AbstractTiledLinear):
  """Tiled mapping with sparse but fixed connectivity.

  There are two additional variable initialization parameters:
  `sparse_indices_sharing_key` and `sparsity_ratio`.

  `sparse_indices_sharing_key` controls which tiles have the same
  connectivity pattern (in the sense of having the same
  tf.SparseTensor.indices). Generally, tiles with the same sharing key
  and `sparsity_ratio` share these indices. There are two special key
  values: `':name:'` and `':shape:'` that get substituted with the
  name and shape of the actual tile, respectively.

  For example, if an LSTM cell maps inputs ('x', 'h') to ('f, 'i, 'j',
  'o'), then the following makes all weight matrices from the input
  'x' to any of the gates or the candidate update share connectivity
  structure. Similarly, there is connectivity pattern sharing between
  weight matrices mapping from the recurrent state 'h'.

      var_init_params=OrderedDict([
          ('W_x_.*', {'sparse_indices_sharing_key': 'x'}),
          ('W_h_.*', {'sparse_indices_sharing_key': 'h'}),
          ('.*', {'sparsity_ratio': 0.5,
                  'initializer': tf.random_uniform_initializer(-1, 1)})
      ])

  Note that only the sparse indices are shared, the values are
  different (unless playing tricks with the 'initializer' param).

  If `sparsity_ratio` is set (to a float number in [0,1]), then this
  represents the proportion of entries non-missing entries in the
  tile. The actual connectivity pattern is determined randomly.

  In the future, there may be support for band and block diagonal
  matrices.
  """

  def __init__(self, input_name_and_sizes, output_name_and_sizes,
               var_init_params=None, name='sparse_tiled_linear'):
    super(SparseTiledLinear, self).__init__(
        input_name_and_sizes, output_name_and_sizes,
        var_init_params=var_init_params, name=name)
    self._sparse_indices_cache = {}
    # Cache the SparseTensor instances to avoid the considerable
    # overhead of creating duplicates just to be optimized out.
    self._sparse_variable_cache = {}

  def _find_or_create_sparse_indices(self, name, shape):
    ratio = self._find_var_init_param(name, 'sparsity_ratio', None)
    assert ratio, 'sparsity_ratio must be specified.'
    sharing_key = self._find_var_init_param(name, 'sparse_indices_sharing_key',
                                            ':name:')
    if sharing_key == ':name:':
      key = name
    if sharing_key == ':shape:':
      sharing_key = shape
    key = (sharing_key, ratio)
    if key not in self._sparse_indices_cache:
      logging.info('Creating sparse indices for %s%r with key %r.',
                   name, shape, key)
      self._sparse_indices_cache[key] = utils.sparse_random_indices(ratio,
                                                                    shape)
    return self._sparse_indices_cache[key]

  def _find_or_create_sparse_variable(self, name, sparse_indices, shape,
                                      initializer=None, partitioner=None,
                                      regularizer=None):
    if name not in self._sparse_variable_cache:
      logging.info('Create sparse variable %s.', name)
      self._sparse_variable_cache[name] = utils.get_sparse_variable(
          name, sparse_indices, shape=shape, initializer=initializer,
          partitioner=partitioner, regularizer=regularizer)
    return self._sparse_variable_cache[name]

  def valid_var_init_param_keys(self):
    return (super(SparseTiledLinear, self).valid_var_init_param_keys() +
            ['sparse_indices_sharing_key', 'sparsity_ratio'])

  def _get_variable(self, name, shape,
                    default_initializer=None, default_partitioner=None,
                    default_regularizer=None, sparse_indices=None):
    initializer = self._find_var_init_param(
        name, 'initializer', default_initializer)
    partitioner = self._find_var_init_param(
        name, 'partitioner', default_partitioner)
    regularizer = self._find_var_init_param(
        name, 'regularizer', default_regularizer)
    sparse_indices = self._find_or_create_sparse_indices(name, shape)
    return self._find_or_create_sparse_variable(
        name, sparse_indices, shape=shape, initializer=initializer,
        partitioner=partitioner, regularizer=regularizer)

  def _build_tiled_linear(self, inputs, input_name_and_sizes,
                          output_name_and_sizes, add_bias):
    results = []
    for output_name, output_size in output_name_and_sizes:
      r = 0.0
      for input_, (input_name, input_size) in zip(inputs, input_name_and_sizes):
        name = 'W_{}_{}'.format(input_name, output_name)
        weight = self._get_variable(
            name, shape=[output_size, input_size])
        r += tf.sparse_tensor_dense_matmul(weight, input_, adjoint_b=True)
      r = tf.transpose(r)
      if add_bias:
        # Biases are dense, hence we call _get_variable of the base
        # class.
        r += super(SparseTiledLinear, self)._get_variable(
            'B_{}'.format(output_name), shape=[output_size],
            default_initializer=tf.zeros_initializer())
      results.append(r)
    return results


# TODO(melisgl): Since computation is the same as in TiledLinear,
# perhaps this should be implemented as a custom getter (see
# tf.get_variable) instead of being tied to tiling.
class OverlaidTiledLinear(TiledLinear):
  """Tiled mapping with weight sharing and low-rank overlays.

  To reduce the number of parameters, one may want to share weight
  matrices. This class makes that sharing possible in the form of W_1
  = s_1*W + a_1*b_1 and W_2 = s_2*W + a_2*b_2 where the s are scalars,
  and a*b are low-rank matrices.

  `overlay_sharing_key` controls which tiles share the same underlying
  weight matrix. Generally, tiles with the same sharing key and 2D
  shape. There are two special key values: `':name:'` and `':shape:'`
  that get substituted with the name and shape of the actual tile,
  respectively.

  For example, if an LSTM cell maps inputs ('x', 'h') to ('f, 'i, 'j',
  'o'), then the following makes all weight matrices from the input
  'x' to any of the gates or the candidate update share the underlying
  full rank weight matrix.

      var_init_params=OrderedDict([
          ('W_x_i', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': 16}),
          ('W_x_j', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': 10}),
          ('W_x_f', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': 8}),
          ('W_x_o', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': 11}),
      ])

  That is, W_x_i = s_W_x_i * W_x_any + a_W_x_i * b_W_x_i where 's_' is
  a scalar, and 'a_', 'b_' are of shape [N, 16], [16, N] respectively.
  W_x_j and the other are computed similarly by adding a low-rank
  overlay ('a_'*'b_') on top of a shared weight matrix ('W_x_any').
  """

  def __init__(self, *args, **kwargs):
    super(OverlaidTiledLinear, self).__init__(*args, **kwargs)
    self._matrix_cache = {}

  def _get_variable(self, name, shape,
                    default_initializer=None, default_partitioner=None,
                    default_regularizer=None):
    if len(shape) != 2:
      return super(OverlaidTiledLinear, self)._get_variable(
          name, shape, default_initializer=default_initializer,
          default_partitioner=default_partitioner,
          default_regularizer=default_regularizer)
    else:
      rank = self._find_var_init_param(name, 'overlay_rank', 0)
      sharing_key = self._find_var_init_param(name, 'overlay_sharing_key',
                                              ':name:')
      if sharing_key == ':name:':
        sharing_key = name
      if sharing_key == ':shape:':
        sharing_key = shape
      if (sharing_key in self._matrix_cache and
          not tf.get_variable_scope().reuse):
        scaler = super(OverlaidTiledLinear, self)._get_variable(
            's_'+name, [shape[1]], default_initializer=tf.ones_initializer())
        base = scaler*self._matrix_cache[sharing_key]
      else:
        base = super(OverlaidTiledLinear, self)._get_variable(
            sharing_key, shape, default_initializer=default_initializer,
            default_partitioner=default_partitioner,
            default_regularizer=default_regularizer)
        self._matrix_cache[sharing_key] = base
      if rank == 0:
        return base
      else:
        overlay = self._low_rank_matrix(name, rank=rank, shape=shape)
        return base+overlay

  def _low_rank_matrix(self, name, rank=None, shape=None,
                       initializer=None, trainable=True):
    assert len(shape) == 2
    a = super(OverlaidTiledLinear, self)._get_variable(
        'a_'+name, [shape[0], rank], default_initializer=initializer)
    b = super(OverlaidTiledLinear, self)._get_variable(
        'b_'+name, [rank, shape[1]], default_initializer=initializer)
    return tf.matmul(a, b)

  def valid_var_init_param_keys(self):
    return (super(OverlaidTiledLinear, self).valid_var_init_param_keys() +
            ['overlay_sharing_key', 'overlay_rank'])
