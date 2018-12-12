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

"""rnn_cell.NASCell adapted to support transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class NASCell(tf.nn.rnn_cell.RNNCell):
  """Neural Architecture Search (NAS) recurrent network cell.

  This implements the recurrent cell from the paper:

    https://arxiv.org/abs/1611.01578

  Barret Zoph and Quoc V. Le.
  "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.

  The class uses an optional projection layer.
  """

  def __init__(self, num_units, num_proj=None,
               use_biases=False, reuse=None,
               initializer=None,
               input_transform=None,
               state_transform=None,
               update_transform=None):
    """Initialize the parameters for a NAS cell.

    Args:
      num_units: int, The number of units in the NAS cell
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      use_biases: (optional) bool, If True then use biases within the cell. This
        is False by default.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      initializer: Initializer for the variables.
      input_transform: None, or a function of one argument that
        massages the input in some way. For example, variational
        dropout can be implemted by passing a Dropout object here.
      state_transform: Similar to input_transform, this is
        applied to the recurrent state.
      update_transform: Similar to input_transform, this is
        applied to the proposed update ('j').
    """
    super(NASCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._num_proj = num_proj
    self._use_biases = use_biases
    self._reuse = reuse

    if num_proj is not None:
      self._state_size = tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj)
      self._output_size = num_proj
    else:
      self._state_size = tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
      self._output_size = num_units
    self._initializer = initializer
    self._input_transform = input_transform
    self._state_transform = state_transform
    assert update_transform is None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of NAS Cell.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: This must be a tuple of state Tensors, both `2-D`, with column
        sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        NAS Cell after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of NAS Cell after reading `inputs`
        when the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = tf.sigmoid
    tanh = tf.tanh
    relu = tf.nn.relu

    num_proj = self._num_units if self._num_proj is None else self._num_proj

    def maybe_transform(transform, x):
      if transform is None:
        return x
      else:
        return transform(x)

    (c_prev, m_prev) = state
    m_prev = maybe_transform(self._state_transform, m_prev)

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    inputs = maybe_transform(self._input_transform, inputs)
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    # Variables for the NAS cell. W_m is all matrices multiplying the
    # hiddenstate and W_inputs is all matrices multiplying the inputs.
    concat_w_m = tf.get_variable(
        "recurrent_kernel", [num_proj, 8 * self._num_units],
        initializer=self._initializer, dtype=dtype)
    concat_w_inputs = tf.get_variable(
        "kernel", [input_size.value, 8 * self._num_units],
        initializer=self._initializer, dtype=dtype)

    m_matrix = tf.matmul(m_prev, concat_w_m)
    inputs_matrix = tf.matmul(inputs, concat_w_inputs)

    if self._use_biases:
      b = tf.get_variable(
          "bias",
          shape=[8 * self._num_units],
          initializer=tf.zeros_initializer(),
          dtype=dtype)
      m_matrix = tf.nn.bias_add(m_matrix, b)

    # The NAS cell branches into 8 different splits for both the hiddenstate
    # and the input
    m_matrix_splits = tf.split(axis=1, num_or_size_splits=8,
                               value=m_matrix)
    inputs_matrix_splits = tf.split(axis=1, num_or_size_splits=8,
                                    value=inputs_matrix)

    # First layer
    layer1_0 = sigmoid(inputs_matrix_splits[0] + m_matrix_splits[0])
    layer1_1 = relu(inputs_matrix_splits[1] + m_matrix_splits[1])
    layer1_2 = sigmoid(inputs_matrix_splits[2] + m_matrix_splits[2])
    layer1_3 = relu(inputs_matrix_splits[3] * m_matrix_splits[3])
    layer1_4 = tanh(inputs_matrix_splits[4] + m_matrix_splits[4])
    layer1_5 = sigmoid(inputs_matrix_splits[5] + m_matrix_splits[5])
    layer1_6 = tanh(inputs_matrix_splits[6] + m_matrix_splits[6])
    layer1_7 = sigmoid(inputs_matrix_splits[7] + m_matrix_splits[7])

    # Second layer
    l2_0 = tanh(layer1_0 * layer1_1)
    l2_1 = tanh(layer1_2 + layer1_3)
    l2_2 = tanh(layer1_4 * layer1_5)
    l2_3 = sigmoid(layer1_6 + layer1_7)

    # Inject the cell
    l2_0 = tanh(l2_0 + c_prev)

    # Third layer
    l3_0_pre = l2_0 * l2_1
    new_c = l3_0_pre  # create new cell
    l3_0 = l3_0_pre
    l3_1 = tanh(l2_2 + l2_3)

    # Final layer
    new_m = tanh(l3_0 * l3_1)

    # Projection layer if specified
    if self._num_proj is not None:
      concat_w_proj = tf.get_variable(
          "projection_weights", [self._num_units, self._num_proj],
          dtype)
      new_m = tf.matmul(new_m, concat_w_proj)

    new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_m)
    return new_m, new_state
