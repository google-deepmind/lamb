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

"""An LSTM cell."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lamb import tiled_linear
from lamb import utils
import six
import tensorflow.compat.v1 as tf


class TiledLSTMCell(tf.nn.rnn_cell.RNNCell):
  """An LSTM cell with tiled connections.

  Supports various connectivity patterns such as the vanilla, dense
  TiledLinear, and also SparseTiledLinear, LayerNormedTiledLinear.
  """

  def __init__(
      self, num_units,
      use_peepholes=False, cell_clip=None,
      initializer=None, num_proj=None,
      feature_mask_rounds=0,
      feature_mask_rank=0,
      tie_gates=False,
      cap_input_gate=True,
      layer_norm=False,
      activation=tf.tanh,
      input_transform=None,
      state_transform=None,
      update_transform=None,
      tiled_linear_class=None,
      tiled_linear_var_init_params=None):
    """Initialize the parameters of a single LSTM layer.

    Args:
      num_units: int, The number of hidden units in the layer.
      use_peepholes: bool, set True to enable diagonal/peephole connections
        (non implemented).
      cell_clip: (optional) A float value, if provided the cell state
        is clipped to be in the [-cell_clip, cell_clip] range prior to
        the cell output activation.
      initializer: (optional) The default initializer to use for the
        weight and projection matrices.
      num_proj: (optional) int, The output size of the non-linear
        transformation (usually `h`) of the cell state (`c`). If None,
        no projection is performed and `h=tanh(c)`. If provided, then
        `h` is `tanh(c)` projected to `num_proj` dimensions.
      feature_mask_rounds: Gate the input and the state before they are used for
        calculating all the other stuff (i.e. i, j, o, f). This allows input
        features to be reweighted based on the state, and state features to be
        reweighted based on the input. When feature_mask_rounds is 0, there is
        no extra gating in the LSTM. When 1<=, the input is gated: x *=
        2*sigmoid(affine(h))). When 2<=, the state is gated: h *=
        2*sigmoid(affine(x))). For higher number of rounds, the alternating
        gating continues.
      feature_mask_rank: If 0, the linear transforms are full rank, dense
        matrices. If >0, then the matrix representing the linear transform is
        factorized as the product of two low rank matrices ([*, rank] and [rank,
        *]). This reduces the number of parameters greatly.
      tie_gates: Whether to make the input gate one minus the forget gate.
      cap_input_gate: Whether to cap the input gate at one minus the
        forget gate (if they are not tied, of course). This ensures
        'c' is in [-1,1] and makes training easier especially in the
        early stages.
      layer_norm: Whether to use Layer Normalization.
      activation: Activation function of the inner states.
      input_transform: None, or a function of one argument that
        massages the input in some way. For example, variational
        dropout can be implemted by passing a Dropout object here.
      state_transform: Similar to input_transform, this is
        applied to the recurrent state.
      update_transform: Similar to input_transform, this is
        applied to the proposed update ('j').
      tiled_linear_class: A class such as tiled_linear.TiledLinear
        that's instantiated an unspecified number of times with the
        same tiled_linear_var_init_params but with possibly different
        inputs and outputs. If layer_norm is false, the default is
        tiled_linear.TiledLinear else it's
        tiled_linear.LayerNormedTiledLinear.
      tiled_linear_var_init_params: Passed right on to
        `tiled_linear_class` as the `var_init_params` argument.
    """
    assert not use_peepholes, 'Peepholes are not implemented in LSTMCell.'
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._feature_mask_rounds = feature_mask_rounds
    self._feature_mask_rank = feature_mask_rank
    self._tie_gates = tie_gates
    self._cap_input_gate = cap_input_gate
    self._layer_norm = layer_norm
    self._activation = activation
    self._input_transform = input_transform
    self._state_transform = state_transform
    self._update_transform = update_transform
    if tiled_linear_class is None:
      if layer_norm:
        tiled_linear_class = tiled_linear.LayerNormedTiledLinear
      else:
        tiled_linear_class = tiled_linear.TiledLinear
    self._tiled_linear_class = tiled_linear_class
    self._tiled_linear_var_init_params = tiled_linear_var_init_params
    self._tiled_linear_mod = None

    if num_proj:
      self._output_size = num_proj
    else:
      self._output_size = num_units

    self._state_size = tf.nn.rnn_cell.LSTMStateTuple(
        num_units, self._output_size)

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @staticmethod
  def _do_feature_masking(x, y, num_x, num_y, rounds, rank):
    for round_ in six.moves.range(rounds):
      # Even rounds correspond to input transforms. Odd rounds to state
      # transforms. Implemented this way because feature_mask_rounds=1 with a
      # single round of transforming the state does not seem to improve things
      # much. Concurrent updates were also tested, but were not an improvement
      # either.
      transforming_x = (round_ % 2 == 0)
      fm_name = 'fm_' + str(round_)
      if rank == 0:  # full rank case
        if transforming_x:
          x *= 2*tf.sigmoid(utils.linear(y, num_x, bias=True, scope=fm_name))
        else:
          y *= 2*tf.sigmoid(utils.linear(x, num_y, bias=True, scope=fm_name))
      else:  # low-rank factorization case
        if transforming_x:
          shape = [num_y, num_x]
        else:
          shape = [num_x, num_y]
        a, b = utils.low_rank_factorization(fm_name + '_weight', shape, rank)
        bias = tf.get_variable(fm_name + '_bias', shape[1],
                               initializer=tf.zeros_initializer())
        if transforming_x:
          x *= 2*tf.sigmoid(tf.matmul(tf.matmul(y, a), b) + bias)
        else:
          y *= 2*tf.sigmoid(tf.matmul(tf.matmul(x, a), b) + bias)
    return x, y

  def __call__(self, input_, state, scope=None):
    """Run one step of LSTM.

    All tensor arguments are shaped [batch_size, *].

    Args:
      input_: A tensor.
      state: An LSTMStateTuple.
      scope: VariableScope for the created subgraph; defaults to
        `LSTMCell`.

    Returns:
      A tuple containing:
      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
        LSTM after one time step.
        Here output_dim is:
           - num_proj if num_proj was set,
           - num_units otherwise.
      - An LSTMStateTuple of Tensors representing the new state
        of the LSTM after one time step.

    Raises:
      ValueError: If input size cannot be inferred from `input_`
      via static shape inference.
    """
    num_units = self._num_units
    num_proj = num_units if self._num_proj is None else self._num_proj
    num_inputs = input_.get_shape().with_rank(2)[1]

    def maybe_transform(transform, x):
      if transform is None:
        return x
      else:
        return transform(x)

    with tf.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):
      # Apply transformations to the input and the recurrent state.
      transformed_input = maybe_transform(self._input_transform, input_)
      transformed_state = maybe_transform(self._state_transform, state.h)

      # Let's transform the input and state further with 'feature masking'.
      transformed_input, transformed_state = self._do_feature_masking(
          transformed_input, transformed_state,
          num_inputs, num_units,
          self._feature_mask_rounds, self._feature_mask_rank)

      inputs = [transformed_input, transformed_state]

      input_name_and_sizes = [('x', num_inputs),
                              ('h', num_proj)]
      output_name_and_sizes = [('j', num_units),
                               ('o', num_units),
                               ('f', num_units)]
      if not self._tie_gates:
        output_name_and_sizes.append(('i', num_units))

      if self._tiled_linear_mod is None:
        self._tiled_linear_mod = self._tiled_linear_class(
            input_name_and_sizes, output_name_and_sizes,
            self._tiled_linear_var_init_params)
      if self._tie_gates:
        j_pre, o_pre, f_pre = self._tiled_linear_mod(inputs)
      else:
        j_pre, o_pre, f_pre, i_pre = self._tiled_linear_mod(inputs)
      # Compute the cell state c.
      f = tf.sigmoid(f_pre)
      j = self._activation(j_pre)
      j = maybe_transform(self._update_transform, j)
      o = tf.sigmoid(o_pre)
      if self._tie_gates:
        c = f * state.c + (1-f) * j
      else:
        i = tf.sigmoid(i_pre)
        if self._cap_input_gate:
          c = f * state.c + tf.minimum(1-f, i) * j
        else:
          c = f * state.c + i * j

      if self._layer_norm:
        c2 = utils.layer_norm(c, [1], scope='ln_c')
      else:
        c2 = c

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type

      h = o * self._activation(c2)
      if self._num_proj is not None:
        h = utils.linear(h, self._num_proj, bias=False, scope='projection')

    return h, tf.nn.rnn_cell.LSTMStateTuple(c, h)
