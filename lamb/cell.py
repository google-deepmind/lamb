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

"""RNN Cell builder."""

# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from absl import logging
import six
import tensorflow.compat.v1 as tf

# pylint: disable=g-bad-import-order
from lamb import utils

from lamb import tiled_linear
from lamb.nascell import NASCell
from lamb.tiled_lstm import TiledLSTMCell
from lamb.tiled_rhn import TiledRHNCell
from lamb.res_multi_rnn_cell import ResMultiRNNCell
from lamb.skip_multi_rnn_cell import SkipMultiRNNCell

from lamb.dropout import DirichletDropout
from lamb.dropout import DriftingDropout
from lamb.dropout import Dropout
from lamb.dropout import GaussianDropout
from tensorflow.contrib import framework as contrib_framework


def build_cell(model, num_layers, hidden_size,
               layer_norm, cell_init_factor,
               shared_mask_dropout,
               input_dropout, inter_layer_dropout, state_dropout,
               update_dropout, state_dropout_flip_rate,
               tie_forget_and_input_gates, cap_input_gate, forget_bias,
               feature_mask_rounds, feature_mask_rank,
               overlay_rank, sparsity_ratio,
               cell_clip, activation_fn,
               lstm_skip_connection, residual_connections):

  cell_initializer = utils.variance_scaling_initializer(
      scale=cell_init_factor, mode='fan_in', distribution='truncated_normal')

  def hidden_size_for_layer(layer_index):
    if isinstance(hidden_size, int):
      return hidden_size
    elif layer_index < len(hidden_size):
      return hidden_size[layer_index]
    else:
      return hidden_size[-1]

  def dropout(dropout_rate, share=shared_mask_dropout,
              flip_prob=None, kind='bernoulli', scaler=1.0):
    if dropout_rate is not None:
      # The same graph is used for training and evaluation with different
      # dropout rates. Passing the constant configured dropout rate here would
      # be a subtle error.
      assert contrib_framework.is_tensor(dropout_rate)
      if flip_prob is not None:
        assert kind == 'bernoulli'
        return DriftingDropout(1-dropout_rate, flip_prob=flip_prob,
                               scaler=scaler)
      elif kind == 'bernoulli':
        return Dropout(1-dropout_rate, share_mask=share, scaler=scaler)
      elif kind == 'dirichlet':
        return DirichletDropout(1-dropout_rate, share_mask=share, scaler=scaler)
      elif kind == 'gaussian':
        return GaussianDropout(1-dropout_rate, share_mask=share, scaler=scaler)
      else:
        assert False

  # We don't use DriftingDropout currently. Ignore it.
  state_dropout_flip_rate = state_dropout_flip_rate

  # Set up input_transforms for the layers based on
  # {input,inter_layer}_dropout.
  input_transforms = []
  for layer_index in six.moves.range(num_layers):
    if model in ['lstm', 'nas']:
      if layer_index == 0:
        transform = dropout(input_dropout)
      elif layer_index > 0:
        transform = dropout(inter_layer_dropout)
      else:
        transform = None
    elif model == 'rhn':
      if layer_index == 0:
        transform = dropout(input_dropout)
      else:
        # The input is not fed to higher layers.
        transform = None
    else:
      assert False
    input_transforms.append(transform)

  # Populate state_transforms to handle state_dropout. This is currently the
  # same for LSTM and RHN: all layers have the same dropout mask, possibly
  # with further sharing over time steps.
  state_transforms = []
  for layer_index in six.moves.range(num_layers):
    transform = dropout(state_dropout, share=True)
    state_transforms.append(transform)

  # Populate update_transforms to handle update_dropout. This is currently the
  # same for LSTM and RHN: all layers have their own dropout mask which may be
  # shared between time steps.
  update_transforms = []
  if model == 'lstm' and (tie_forget_and_input_gates or cap_input_gate):
    # The 1.5 is to reach a more non-linear part of the output tanh.
    base_scale = 1.5
  else:
    base_scale = 1.0
  for layer_index in six.moves.range(num_layers):
    if update_dropout is None:
      scaler = 1.0
    else:
      scaler = base_scale*(1-update_dropout)
    update_transforms.append(dropout(
        update_dropout,
        # Dropout mask for the recurrent state needs to be the
        # same for all time steps.
        share=True,
        # This makes update dropout do mask*x at training time and
        # x*(1-r) at test time instead of usual mask*x/(1-r) and
        # x, respectively.
        scaler=scaler))

  def make_lstm_column():
    init_params = collections.OrderedDict([
        ('B_f', {'initializer': utils.variance_scaling_initializer(
            scale=cell_init_factor, distribution='truncated_normal',
            mean=forget_bias)})
    ])
    if overlay_rank > 0:
      assert sparsity_ratio < 0
      # TODO(melisgl): Specify initializers for the shared matrices.
      tiled_linear_class = tiled_linear.OverlayedTiledLinear
      init_params.update(collections.OrderedDict([
          ('W_x_i', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_x_j', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_x_f', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_x_o', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_h_i', {'overlay_sharing_key': 'W_h_any',
                     'overlay_rank': overlay_rank}),
          ('W_h_j', {'overlay_sharing_key': 'W_h_any',
                     'overlay_rank': overlay_rank}),
          ('W_h_f', {'overlay_sharing_key': 'W_h_any',
                     'overlay_rank': overlay_rank}),
          ('W_h_o', {'overlay_sharing_key': 'W_h_any',
                     'overlay_rank': overlay_rank}),
      ]))
    elif sparsity_ratio >= 0.0:
      assert overlay_rank == -1
      tiled_linear_class = tiled_linear.SparseTiledLinear
      # This is equivalent to using cell_initializer scaled by
      # 1/sparsity_ratio.
      sparse_initializer = tf.truncated_normal_initializer(
          stddev=math.sqrt(cell_init_factor /
                           sparsity_ratio /
                           # TODO(melisgl): This is off if the input
                           # embedding size is different from the hidden
                           # size.
                           hidden_size))
      init_params.update(collections.OrderedDict([
          ('W_x_.*', {'sparse_indices_sharing_key': 'W_x'}),
          ('W_h_.*', {'sparse_indices_sharing_key': 'W_h'}),
          ('W_x', {'sparsity_ratio': sparsity_ratio,
                   'initializer': sparse_initializer}),
          ('W_h', {'sparsity_ratio': sparsity_ratio,
                   'initializer': sparse_initializer}),
      ]))
    else:
      if layer_norm:
        tiled_linear_class = tiled_linear.LayerNormedTiledLinear
      else:
        tiled_linear_class = tiled_linear.TiledLinear
      init_params.update(collections.OrderedDict([
          ('W_.*', {'initializer': cell_initializer}),
          ('B_.*', {'initializer': cell_initializer})
      ]))
    def make_layer(layer_index):
      cell = TiledLSTMCell(
          hidden_size_for_layer(layer_index),
          tie_gates=tie_forget_and_input_gates,
          cap_input_gate=cap_input_gate,
          feature_mask_rounds=feature_mask_rounds,
          feature_mask_rank=feature_mask_rank,
          input_transform=input_transforms[layer_index],
          state_transform=state_transforms[layer_index],
          update_transform=update_transforms[layer_index],
          tiled_linear_class=tiled_linear_class,
          tiled_linear_var_init_params=init_params,
          initializer=cell_initializer,
          cell_clip=cell_clip if cell_clip > 0 else None,
          layer_norm=layer_norm,
          activation=eval(activation_fn))  # pylint: disable=eval-used
      return cell
    layers = [make_layer(i) for i in six.moves.range(num_layers)]
    if lstm_skip_connection:
      assert not residual_connections
      return SkipMultiRNNCell(layers)
    elif residual_connections:
      return ResMultiRNNCell(layers)
    else:
      return tf.nn.rnn_cell.MultiRNNCell(layers)

  def make_rhn_column():
    init_params = collections.OrderedDict([
        ('B_c', {'initializer': tf.constant_initializer(forget_bias)}),
    ])

    if overlay_rank > 0:
      assert sparsity_ratio < 0
      # TODO(melisgl): Specify initializers for the shared matrices.
      tiled_linear_class = tiled_linear.OverlayedTiledLinear
      init_params.update(collections.OrderedDict([
          ('W_x_h', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_x_c', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_x_t', {'overlay_sharing_key': 'W_x_any',
                     'overlay_rank': overlay_rank}),
          ('W_s_h', {'overlay_sharing_key': 'W_s_any',
                     'overlay_rank': overlay_rank}),
          ('W_s_c', {'overlay_sharing_key': 'W_s_any',
                     'overlay_rank': overlay_rank}),
          ('W_s_t', {'overlay_sharing_key': 'W_s_any',
                     'overlay_rank': overlay_rank}),
      ]))
    elif sparsity_ratio >= 0.0:
      assert overlay_rank == -1
      tiled_linear_class = tiled_linear.SparseTiledLinear
      sparse_initializer = tf.truncated_normal_initializer(
          stddev=math.sqrt(cell_init_factor /
                           sparsity_ratio /
                           # TODO(melisgl): This is off if the input
                           # embedding size is different from the hidden
                           # size.
                           hidden_size))
      init_params.update(collections.OrderedDict([
          ('W_x_.*', {'sparse_indices_sharing_key': 'W_x'}),
          ('W_s_.*', {'sparse_indices_sharing_key': 'W_s'}),
          ('W_x', {'sparsity_ratio': sparsity_ratio,
                   'initializer': sparse_initializer}),
          ('W_s', {'sparsity_ratio': sparsity_ratio,
                   'initializer': sparse_initializer}),
      ]))
    else:
      tiled_linear_class = tiled_linear.TiledLinear
      init_params.update(collections.OrderedDict([
          ('W_.*', {'initializer': cell_initializer}),
      ]))
    logging.info('Creating RHN of depth %s', num_layers)
    if layer_norm:
      logging.warn('RHN does not support layer normalization.')
    cell = TiledRHNCell(
        hidden_size,
        depth=num_layers,
        tie_gates=tie_forget_and_input_gates,
        input_transform=input_transforms[layer_index],
        state_transform=state_transforms[layer_index],
        update_transform=update_transforms[layer_index],
        tiled_linear_class=tiled_linear_class,
        tiled_linear_var_init_params=init_params,
        cell_clip=cell_clip if cell_clip > 0 else None,
        activation=eval(activation_fn))  # pylint: disable=eval-used
    return cell

  def make_nas_column():
    assert not layer_norm
    def make_layer(layer_index):
      logging.info('Creating layer %s', layer_index)
      cell = NASCell(
          hidden_size,
          input_transform=input_transforms[layer_index],
          state_transform=state_transforms[layer_index],
          update_transform=update_transforms[layer_index],
          initializer=cell_initializer)
      return cell
    layers = [make_layer(i) for i in six.moves.range(num_layers)]
    if lstm_skip_connection:
      assert not residual_connections
      return SkipMultiRNNCell(layers)
    elif residual_connections:
      return ResMultiRNNCell(layers)
    else:
      return tf.nn.rnn_cell.MultiRNNCell(layers)

  assert len(hidden_size) <= num_layers
  if model == 'lstm':
    return make_lstm_column()
  elif model == 'rhn':
    assert len(set(hidden_size)) == 1
    hidden_size = hidden_size[0]
    return make_rhn_column()
  elif model == 'nas':
    assert len(set(hidden_size)) == 1
    hidden_size = hidden_size[0]
    return make_nas_column()
  else:
    assert False
