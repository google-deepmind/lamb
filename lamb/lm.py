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

"""A simple language model."""

# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from lamb import utils
from lamb.cell import build_cell
from lamb.dropout import Dropout
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class LM(object):
  """A language model."""

  def __init__(self, config, _model_only=False):
    """A language model.

    Args:
      config: A dictionary with the configuration options (see README.md).
      _model_only: For internal use only.
    """
    if not _model_only:
      logging.info('Finalized parameters to follow.')
      logging.info('%s', str(config))
      logging.info('Building model.')
      self._build_model(config)
      logging.info('Building loss.')
      self._build_loss(config)
      self._check_budget(config)
      self.config = config
    else:
      self._build_model(config)

  @staticmethod
  def num_params(config):
    g = tf.Graph()
    with g.as_default() as g:
      # Speed graph creation up by only expanding the RNN to one step. This
      # graph will be discarded anyway.
      config.max_time_steps = 1
      try:
        LM(config, _model_only=True)
      except (tf.errors.ResourceExhaustedError,
              # Some OOM conditions turn into internal errors.
              tf.errors.InternalError):
        return None
      n = utils.count_trainables()
      return n

  def _build_model(self, config):
    self.global_step_var = tf.Variable(
        tf.zeros([], tf.int64), name='global_step', trainable=False)
    self.learning_rate = tf.placeholder(
        tf.float32, shape=[], name='learning_rate')

    ## Input variables

    self.num_samples = tf.placeholder_with_default(
        1, shape=[], name='num_samples')

    # For MT, this is the source language text. For LM, this is not used.
    if config.conditioning_separator:
      assert config.episodic, 'conditioning and non-episodic do not mix.'
      self.conditioning = tf.placeholder(
          dtype=tf.int64, shape=[config.max_time_steps, None],
          name='conditioning')
      self.conditioning_len = tf.placeholder(dtype=tf.int64, shape=[None],
                                             name='conditioning_len')

    # For plain LM, this is the input text. For MT this is the target language
    # text.
    self.source = tf.placeholder(
        dtype=tf.int64, shape=[config.max_time_steps, None], name='source')
    self.source_len = tf.placeholder(dtype=tf.int64, shape=[None],
                                     name='source_len')

    # This is the ground truth text to be predicted. A shifted by one version
    # version of self.source.
    self.target = tf.placeholder(
        dtype=tf.int64, shape=[config.max_time_steps, None], name='target')

    def maybe_create_dropout_placeholder(configured_dropout_rate, name):
      if configured_dropout_rate > 0.0:
        return tf.placeholder(tf.float32, shape=[], name=name)
      else:
        return None

    self.embedding_dropout = maybe_create_dropout_placeholder(
        config.embedding_dropout, 'embedding_dropout')
    self.token_dropout = maybe_create_dropout_placeholder(
        config.token_dropout, 'token_dropout')
    self.input_dropout = maybe_create_dropout_placeholder(
        config.input_dropout, 'input_dropout')
    self.inter_layer_dropout = maybe_create_dropout_placeholder(
        config.inter_layer_dropout, 'inter_layer_dropout')
    self.update_dropout = maybe_create_dropout_placeholder(
        config.update_dropout, 'update_dropout')
    self.state_dropout = maybe_create_dropout_placeholder(
        config.state_dropout, 'state_dropout')
    self.flip_prob = maybe_create_dropout_placeholder(
        config.state_dropout_flip_rate, 'flip_prob')
    self.output_dropout = maybe_create_dropout_placeholder(
        config.output_dropout, 'output_dropout')
    self.downprojected_output_dropout = maybe_create_dropout_placeholder(
        config.downprojected_output_dropout, 'downprojected_output_dropout')

    self.softmax_temperature = tf.placeholder_with_default(
        1.0, shape=[], name='softmax_temperature')

    ## Training

    embedding_initializer = tf.variance_scaling_initializer(
        scale=config.embedding_init_factor, mode='fan_out',
        distribution='truncated_normal')
    output_initializer = tf.variance_scaling_initializer(
        scale=config.output_init_factor, mode='fan_in',
        distribution='truncated_normal')
    batch_size = tf.shape(self.source)[1]

    last_hidden_size = utils.ensure_list(config.hidden_size)[-1]

    tb_h = tf.stack([config.max_time_steps*batch_size, last_hidden_size])
    t_b_v = tf.stack([config.max_time_steps, batch_size, config.vocab_size])
    t_bk_o = tf.stack([
        config.max_time_steps,
        batch_size*(config.mos_num_components or 1),
        config.output_embedding_size])
    tbk_o = tf.stack([
        config.max_time_steps*
        batch_size*(config.mos_num_components or 1),
        config.output_embedding_size])
    t_b0_s_v = tf.stack(
        [config.max_time_steps, tf.div(batch_size, self.num_samples),
         self.num_samples, config.vocab_size])

    if config.embed_once:
      with tf.variable_scope('im', initializer=embedding_initializer):
        embedding = tf.get_variable(
            'embedding', [config.vocab_size, config.input_embedding_size],
            initializer=embedding_initializer, dtype=tf.float32)
        if self.embedding_dropout is not None:
          embedding = tf.nn.dropout(
              embedding, 1-self.embedding_dropout,
              noise_shape=tf.stack([config.vocab_size, 1]))
        embedded_source = tf.nn.embedding_lookup(embedding, self.source)
        if self.token_dropout is not None:
          embedding = tf.nn.dropout(
              embedding, 1-self.token_dropout,
              noise_shape=tf.stack([config.max_time_steps, batch_size, 1]))
        if config.scale_input_embeddings:
          embedded_source *= tf.sqrt(tf.cast(config.input_embedding_size,
                                             tf.float32))
        sources = embedded_source
    else:
      assert self.embedding_dropout is None, 'Not implemented.'
      assert self.token_dropout is None, 'Not implemented.'
      sources = self.source

    def lm_1(cell, initial_state, inputs, input_lens, scope=None):
      # According to tests (2019-03-13) swap_memory carries only a very penalty
      # so we use it to choose between dynamic_rnn and static_rnn. For some
      # reason, static_rnn can be 2x faster ... sometimes. On the other hand,
      # dynamic_rnn handles memory better even without swap_memory=True.
      if FLAGS.swap_memory:
        return tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                                 time_major=True,
                                 sequence_length=input_lens,
                                 initial_state=initial_state,
                                 swap_memory=FLAGS.swap_memory,
                                 dtype=tf.float32, scope=scope)
      else:
        return tf.nn.static_rnn(cell=cell, inputs=tf.unstack(inputs),
                                sequence_length=input_lens,
                                initial_state=initial_state,
                                dtype=tf.float32, scope=scope)

    # This is for the config.output_once=True case.
    def output_module_1(outputs):
      with tf.variable_scope('om', initializer=output_initializer):
        # Create the matrix and bias for the final projection into the softmax.
        if config.share_input_and_output_embeddings:
          assert config.embed_once, 'Not implemented.'
          softmax_weights = embedding
          softmax_weights_transpose = True
        else:
          softmax_weights = tf.get_variable(
              'weights', [config.output_embedding_size, config.vocab_size],
              dtype=tf.float32)
          softmax_weights_transpose = False
        softmax_bias = tf.get_variable('bias', [1, config.vocab_size],
                                       initializer=tf.zeros_initializer(),
                                       dtype=tf.float32)
        def to_softmax(x, dropout=self.downprojected_output_dropout):
          if dropout is not None:
            if not config.shared_mask_dropout:
              x = tf.nn.dropout(x, 1.0-dropout)
            else:
              x = tf.reshape(x, t_bk_o)
              x = tf.nn.dropout(
                  x, 1.0-dropout,
                  # same mask for all time steps
                  noise_shape=[
                      1, batch_size*(config.mos_num_components or 1),
                      config.output_embedding_size])
              x = tf.reshape(x, tbk_o)
          return (
              self.softmax_temperature*
              (tf.matmul(x, softmax_weights,
                         transpose_b=softmax_weights_transpose) + softmax_bias))

        last_hidden_size = utils.ensure_list(config.hidden_size)[-1]
        outputs_t_b_h = tf.convert_to_tensor(outputs)
        if self.output_dropout is not None:
          if not config.shared_mask_dropout:
            outputs_t_b_h = tf.nn.dropout(
                outputs_t_b_h, 1.0-self.output_dropout)
          else:
            outputs_t_b_h = tf.nn.dropout(
                outputs_t_b_h, 1.0-self.output_dropout,
                noise_shape=[1, batch_size, last_hidden_size])
        outputs_tb_h = tf.reshape(outputs_t_b_h, tb_h)

        if config.mos_num_components == 0:
          if config.output_embedding_size == last_hidden_size:
            return (tf.reshape(to_softmax(outputs_tb_h, None), t_b_v),
                    outputs_t_b_h)
          else:
            downprojected_outputs_tb_o = utils.linear(
                outputs_tb_h, config.output_embedding_size, False,
                initializer=utils.orthogonal_initializer(), scope='projection')
            logits_tb_v = to_softmax(downprojected_outputs_tb_o)
            return tf.reshape(logits_tb_v, t_b_v), outputs_t_b_h
        else:
          logits_tb_v = utils.mixture_of_softmaxes(
              outputs_tb_h, config.mos_num_components,
              config.output_embedding_size, to_softmax)
          return tf.reshape(logits_tb_v, t_b_v), outputs_t_b_h

    # This is for the config.output_once=False case.
    def output_module_per_step_1(outputs_b_h):
      with tf.variable_scope('om', initializer=output_initializer):
        def to_softmax(x, dropout=self.downprojected_output_dropout):
          # Create the matrix and bias for the final projection into the
          # softmax.
          if config.share_input_and_output_embeddings:
            assert config.embed_once, 'Not implemented.'
            softmax_weights = embedding
            softmax_weights_transpose = True
          else:
            softmax_weights = tf.get_variable(
                'weights', [config.output_embedding_size, config.vocab_size],
                dtype=tf.float32)
            softmax_weights_transpose = False
          softmax_bias = tf.get_variable('bias', [1, config.vocab_size],
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
          if dropout is not None:
            x = Dropout(1.0-dropout, share_mask=config.shared_mask_dropout)(x)
          return (self.softmax_temperature *
                  (tf.matmul(x, softmax_weights,
                             transpose_b=softmax_weights_transpose) +
                   softmax_bias))

        last_hidden_size = utils.ensure_list(config.hidden_size)[-1]
        outputs_b_h = Dropout(1.0-self.output_dropout,
                              share_mask=self.output_dropout)(outputs_b_h)

        if config.mos_num_components == 0:
          if config.output_embedding_size == last_hidden_size:
            return to_softmax(outputs_b_h, None)
          else:
            downprojected_outputs_b_o = utils.linear(
                outputs_b_h, config.output_embedding_size, False,
                initializer=utils.orthogonal_initializer(), scope='projection')
            logits_b_v = to_softmax(downprojected_outputs_b_o)
            return logits_b_v
        else:
          logits_b_v = utils.mixture_of_softmaxes(
              outputs_b_h, config.mos_num_components,
              config.output_embedding_size, to_softmax)
          return logits_b_v

    lm = tf.make_template('lm', lm_1)

    def make_cell():
      return build_cell(
          model=config.model,
          num_layers=config.num_layers,
          hidden_size=config.hidden_size,
          layer_norm=config.layer_norm,
          cell_init_factor=config.cell_init_factor,
          shared_mask_dropout=config.shared_mask_dropout,
          input_dropout=self.input_dropout,
          inter_layer_dropout=self.inter_layer_dropout,
          state_dropout=self.state_dropout,
          update_dropout=self.update_dropout,
          state_dropout_flip_rate=self.flip_prob,
          tie_forget_and_input_gates=config.tie_forget_and_input_gates,
          cap_input_gate=config.cap_input_gate,
          forget_bias=config.forget_bias,
          feature_mask_rounds=config.feature_mask_rounds,
          feature_mask_rank=config.feature_mask_rank,
          overlay_rank=config.overlay_rank,
          sparsity_ratio=config.sparsity_ratio,
          cell_clip=config.cell_clip,
          activation_fn=config.activation_fn,
          lstm_skip_connection=config.lstm_skip_connection,
          residual_connections=config.residual_connections)

    def make_conditioning():
      if config.embed_once:
        with tf.variable_scope('cond_im', initializer=embedding_initializer):
          embedding = tf.get_variable(
              'embedding', [config.conditioning_vocab_size,
                            config.input_embedding_size],
              initializer=embedding_initializer, dtype=tf.float32)
          if self.embedding_dropout is not None:
            embedding = tf.nn.dropout(
                embedding, 1-self.embedding_dropout,
                noise_shape=tf.stack([config.conditioning_vocab_size, 1]))
          embedded_source = tf.nn.embedding_lookup(embedding, self.conditioning)
          if self.token_dropout is not None:
            embedding = tf.nn.dropout(
                embedding, 1-self.token_dropout,
                noise_shape=tf.stack([config.max_time_steps, batch_size, 1]))
          if config.scale_input_embeddings:
            embedded_source *= tf.sqrt(tf.cast(config.input_embedding_size,
                                               tf.float32))
          conditioning_sources = embedded_source
      else:
        assert False, 'Not implemented.'

      conditioning_cell = make_cell()
      conditioning_lm = tf.make_template('cond_lm', lm_1)
      initial_state = conditioning_cell.zero_state(batch_size, dtype=tf.float32)
      _, conditioning_last_state = conditioning_lm(
          conditioning_cell, initial_state,
          conditioning_sources, self.conditioning_len)
      return conditioning_last_state

    cell = make_cell()
    if not config.embed_once:
      cell = tf.nn.rnn_cell.EmbeddingWrapper(
          cell, config.vocab_size, config.input_embedding_size,
          initializer=embedding_initializer)
    if config.conditioning_separator:
      self.initial_state = make_conditioning()
    elif config.trainable_initial_state:
      with tf.variable_scope('lm_init'):
        self.initial_state = utils.trainable_initial_state(
            batch_size, cell.state_size)
    else:
      self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, self.last_state = lm(
        cell, self.initial_state, sources, self.source_len)
    self.cell_outputs = tf.convert_to_tensor(outputs)

    if config.output_once:
      output_module = tf.make_template('om', output_module_1)
      logits_, self.dropped_cell_outputs = output_module(outputs)
    else:
      assert config.activation_norm_penalty == 0.0, (
          'activation_norm_penalty not implemented for output_once=False.')
      output_module_per_step = tf.make_template('om', output_module_per_step_1)
      # KLUDGE: calling output_module_per_step here gets rid of the
      # 'rnn/FNCell/' prefix on the variables names so output_once=False and
      # output_once=True checkpoints are compatible.
      output_module_per_step(outputs[0])
      output_cell = utils.FNCell(output_module_per_step, config.vocab_size)
      logits_, _ = tf.nn.dynamic_rnn(cell=output_cell,
                                     inputs=tf.convert_to_tensor(outputs),
                                     time_major=True,
                                     sequence_length=self.source_len,
                                     swap_memory=FLAGS.swap_memory,
                                     dtype=tf.float32)

    def average_samples():
      # logits has shape t_b_v, where b=b0*num_samples. Separate out
      # the samples in a new dimension.
      logits = tf.reshape(logits_, t_b0_s_v)
      if config.model_average == 'geometric':
        x = tf.reduce_sum(logits, axis=2, keepdims=True)
      elif config.model_average == 'arithmetic':
        log_probs = tf.nn.log_softmax(logits)
        x = tf.reduce_logsumexp(log_probs, axis=2, keepdims=True)
      else:
        assert False, 'Not implemented.'
      # x is t_b0_1_v, tile it to t_b0_s_v.
      x = tf.ones_like(logits) * x
      return tf.reshape(x, t_b_v)

    self.logits = tf.cond(tf.equal(self.num_samples, 1),
                          lambda: logits_,
                          average_samples)

  def _build_loss(self, config):

    # Single sample loss (in terms of num_training_samples)
    self.xe_losses = utils.seq_softmax_cross_entropy_with_logits(
        self.logits, self.target, self.source_len,
        config.max_time_steps, reduce_sum=False, name='lm_loss')
    self.xe_loss = tf.reduce_sum(self.xe_losses, axis=0)
    self.log_probs = tf.nn.log_softmax(self.logits)

    if config.l2_penalty == 0.0:
      self.l2_loss = 0.0
    else:
      self.l2_loss = tf.add_n(
          [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    if config.l1_penalty == 0.0:
      self.l1_loss = 0.0
    else:
      self.l1_loss = tf.add_n(
          [tf.reduce_sum(tf.abs(var)) for var in tf.trainable_variables()])
    if config.activation_norm_penalty == 0.0:
      self.activation_norm_loss = 0.0
    else:
      self.activation_norm_loss = tf.reduce_mean(
          # Sum over time to make values compatible with AWD-LSTM by Merity.
          tf.reduce_sum(tf.square(self.dropped_cell_outputs), axis=0))

    self.unregularized_loss = tf.reduce_mean(self.xe_loss)
    self.loss = (self.unregularized_loss +
                 config.l2_penalty * self.l2_loss +
                 config.l1_penalty * self.l1_loss +
                 config.activation_norm_penalty * self.activation_norm_loss)

    def get_scopes_to_train():
      scopes_to_train = ['lm', 'om']
      if config.trainable_initial_state:
        scopes_to_train = ['lm_init'] + scopes_to_train
      if config.embed_once:
        scopes_to_train = ['im'] + scopes_to_train
      if config.conditioning_separator:
        scopes_to_train = ['cond_im', 'cond_lm'] + scopes_to_train
      return scopes_to_train

    def maybe_clip_grads(grads_and_vars):
      logging.info('adding grad norm clipping')
      return utils.clip_gradients_in_scope(
          grads_and_vars, [''], config.max_grad_norm)

    optimizer_builder = utils.get_optimizer(config.optimizer_type)
    optimizer = optimizer_builder(self.learning_rate, config)
    scopes_to_train = get_scopes_to_train()
    grads_and_vars, training_summaries = utils.create_grads(
        optimizer, self.loss, scopes_to_train)
    # For dyneval.
    self.clipped_grads_and_vars = maybe_clip_grads(grads_and_vars)
    # Single minibatch training update
    self.training_update = optimizer.apply_gradients(
        self.clipped_grads_and_vars, global_step=self.global_step_var)
    self.training_summary = tf.summary.merge(
        training_summaries + utils.summaries_for_trainables())

    # Accumulation of gradients across minibatches
    if config.accum_batch_size > -1:
      trained_vars = [var for _, var in grads_and_vars]
      grad_accumulators = [
          tf.Variable(tf.zeros_like(trained_var.initialized_value()),
                      trainable=False)
          for trained_var in trained_vars]
      self.accumulate_grads = tf.group(*[
          accumulator.assign_add(grads_and_vars[0])
          for accumulator, grads_and_vars
          in zip(grad_accumulators, grads_and_vars)])
      accumulated_grads_and_vars = zip(grad_accumulators, trained_vars)
      self.accumulated_training_update = optimizer.apply_gradients(
          maybe_clip_grads(accumulated_grads_and_vars),
          global_step=self.global_step_var)
      # Zero the accumulators after the update.
      with tf.control_dependencies([self.accumulated_training_update]):
        self.accumulated_training_update = tf.group(
            *[var.assign(tf.zeros_like(var)) for var in grad_accumulators])

    logging.info('Model: adding loss gradients finished.')

  def _check_budget(self, config):
    num_trainables = utils.log_trainables()
    if config.num_params > -1:
      assert num_trainables <= config.num_params, (
          'The number of trainable parameters ({}) exceeds the budget ({}). '
          .format(num_trainables, config.num_params))
      if num_trainables < 0.98*(config.num_params-500):
        logging.warn('Number of parameters (%s) is way below the budget (%s)',
                     num_trainables, config.num_params)

  def global_step(self, session=None):
    if session is None:
      session = tf.get_default_session()
    return session.run(self.global_step_var)

  def add_input_to_feed(self, feed, cond, cond_len, source, source_len, target):
    if self.config.conditioning_separator:
      feed.update({self.conditioning: cond,
                   self.conditioning_len: cond_len})
    else:
      assert cond is None
      assert cond_len is None
    feed.update({self.source: source,
                 self.source_len: source_len,
                 self.target: target})
    return feed

  def add_dropout_to_feed(self, feed, multiplier=1):
    config = self.config
    if self.embedding_dropout is not None:
      feed.update({self.embedding_dropout: multiplier*config.embedding_dropout})
    if self.token_dropout is not None:
      feed.update({self.token_dropout: multiplier*config.token_dropout})
    if self.input_dropout is not None:
      feed.update({self.input_dropout: multiplier*config.input_dropout})
    if self.inter_layer_dropout is not None:
      feed.update({self.inter_layer_dropout:
                   multiplier*config.inter_layer_dropout})
    if self.update_dropout is not None:
      feed.update({self.update_dropout: multiplier*config.update_dropout})
    if self.state_dropout is not None:
      feed.update({self.state_dropout: multiplier*config.state_dropout})
    if self.flip_prob is not None:
      feed.update({self.flip_prob: multiplier*config.state_dropout_flip_rate})
    if self.output_dropout is not None:
      feed.update({self.output_dropout: multiplier*config.output_dropout})
    if self.downprojected_output_dropout is not None:
      feed.update({self.downprojected_output_dropout:
                   multiplier*config.downprojected_output_dropout})
    return feed

  def fit(self, feed, session=None):
    """Training step for observed source language example."""
    if session is None:
      session = tf.get_default_session()
    run_options = tf.RunOptions(
        report_tensor_allocations_upon_oom=True)
    _, cost, summary, last_state = session.run(
        [self.training_update, self.unregularized_loss, self.training_summary,
         self.last_state],
        feed_dict=feed, options=run_options)
    return cost, summary, last_state

  def accumulate_gradients(self, feed, session=None):
    if session is None:
      session = tf.get_default_session()
    _, cost, summary, last_state = session.run(
        [self.accumulate_grads, self.unregularized_loss,
         self.training_summary, self.last_state],
        feed_dict=feed)
    return cost, summary, last_state

  def fit_accumulated(self, feed, session=None):
    """Training step for observed source language example."""
    if session is None:
      session = tf.get_default_session()
    session.run([self.accumulated_training_update], feed_dict=feed)
