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

"""An RHN cell."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from lamb import tiled_linear
import six
import tensorflow.compat.v1 as tf


# The state of an RHN cell consists of:
#
# - `s`, the cell state (like TiledLSTMCellState.c)
_TiledRHNStateTuple = collections.namedtuple('TiledRHNStateTuple', ('s'))


class TiledRHNStateTuple(_TiledRHNStateTuple):
  __slots__ = ()

  @property
  def dtype(self):
    s, = self
    return s.dtype


class TiledRHNCell(tf.nn.rnn_cell.RNNCell):
  """An RHN cell with tiled connections.

  A RHN cell is like a simplified LSTM with multiple layers. See
  'Recurrent Highway Networks' by Zilly et al.
  (https://arxiv.org/abs/1607.03474). This implementation is based on
  v3 of that paper.

  Supports various connectivity patterns such as the vanilla, dense
  TiledLinear, and also SparseTiledLinear, LayerNormedTiledLinear.
  """

  def __init__(
      self, num_units, depth=1,
      use_peepholes=False, cell_clip=None,
      initializer=None,
      tie_gates=False,
      activation=tf.tanh,
      input_transform=None,
      state_transform=None,
      update_transform=None,
      tiled_linear_class=None,
      tiled_linear_var_init_params=None):
    """Initialize the parameters of a single RHN layer.

    Args:
      num_units: int, The number of hidden units in the layer.
      depth: int, The number of layers.
      use_peepholes: bool, set True to enable diagonal/peephole connections
        (non implemented).
      cell_clip: (optional) A float value, if provided the cell state
        is clipped to be in the [-cell_clip, cell_clip] range prior to
        the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      tie_gates: Whether to make the input gate one minus the forget gate.
      activation: Activation function of the inner states.
      input_transform: None, or a function of one argument that
        massages the input in some way. For example, variational
        dropout can be implemted by passing a Dropout object here.
      state_transform: Similar to input_transform, this is
        applied to the recurrent state.
      update_transform: Similar to input_transform, this is
        applied to the proposed update ('h').
      tiled_linear_class: A class such as tiled_linear.TiledLinear
        that's instantiated an unspecified number of times with the
        same tiled_linear_var_init_params but with possibly different
        inputs and outputs. Defaults to tiled_linear.TiledLinear.
      tiled_linear_var_init_params: Passed right on to
        `tiled_linear_class` as the `var_init_params` argument.
    """
    assert not use_peepholes, 'Peepholes are not implemented in RHNCell.'
    self._num_units = num_units
    self._depth = depth
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._tie_gates = tie_gates
    self._activation = activation
    self._input_transform = input_transform
    self._state_transform = state_transform
    self._update_transform = update_transform
    if tiled_linear_class is None:
      tiled_linear_class = tiled_linear.TiledLinear
    self._tiled_linear_class = tiled_linear_class
    self._tiled_linear_var_init_params = tiled_linear_var_init_params
    self._tiled_linear_mods = [None]*depth

    self._output_size = num_units
    self._state_size = TiledRHNStateTuple(num_units)

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, input_, state, scope=None):
    """Run one step of RHN.

    All tensor arguments are shaped [batch_size, *].

    Args:
      input_: A tensor.
      state: An TiledRHNStateTuple.
      scope: VariableScope for the created subgraph; defaults to
        `TiledRHNCell`.

    Returns:
      A tuple containing:
      - A `2-D, [batch, num_units]`, Tensor representing the output of
        the RHN after one time step (which consists of `depth` number
        of computational steps).
      - An TiledRHNStateTuple tuple of Tensors representing the new state
        of the RHN after one time step.

    Raises:
      ValueError: If input size cannot be inferred from `input_`
      via static shape inference.
    """
    num_units = self._num_units

    def maybe_transform(transform, x):
      if transform is None:
        return x
      else:
        return transform(x)

    # Apply transformations to the input and the recurrent state.
    transformed_input = maybe_transform(self._input_transform, input_)

    # Let's figure out what the outputs are.
    output_name_and_sizes = [
        # This is the proposed update (usually 'j' in an LSTM).
        ('h', num_units),
        # Called 'carry' gate in the paper. This pretty much plays the
        # part of the forget gate of an LSTM.
        ('c', num_units)]
    if not self._tie_gates:
      # Called 'transform' gate, this is like the input gate of an
      # LSTM.
      output_name_and_sizes.append(('t', num_units))

    with tf.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):
      s = state.s
      for level in six.moves.range(self._depth):
        with tf.variable_scope('layer{}'.format(level)):
          transformed_s = maybe_transform(self._state_transform, s)
          if level == 0:
            inputs = [transformed_input, transformed_s]
            input_name_and_sizes = [
                ('x', transformed_input.get_shape().with_rank(2)[1]),
                # This is the raw cell state. Unlike in an LSTM this
                # is not passed through any non-linearity.
                ('s', num_units)]
          else:
            inputs = [transformed_s]
            input_name_and_sizes = [('s', num_units)]
          if self._tiled_linear_mods[level] is None:
            self._tiled_linear_mods[level] = self._tiled_linear_class(
                input_name_and_sizes, output_name_and_sizes,
                self._tiled_linear_var_init_params)
          if self._tie_gates:
            h_pre, c_pre = self._tiled_linear_mods[level](inputs)
          else:
            h_pre, c_pre, t_pre = self._tiled_linear_mods[level](inputs)
          # Compute the cell state s.
          c = tf.sigmoid(c_pre)
          h = self._activation(h_pre)
          h = maybe_transform(self._update_transform, h)
          if self._tie_gates:
            t = 1 - c
          else:
            t = tf.sigmoid(t_pre)
          s = c * s + t * h

          if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            s = tf.clip_by_value(s, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type

    return s, TiledRHNStateTuple(s)
