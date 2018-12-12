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

"""The training loop."""

# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

from absl import flags
from absl import logging
from lamb import corpus
from lamb import evaluation
from lamb import lamb_flags
from lamb import lm
from lamb import monitoring
from lamb import utils
from lamb.averaged import Averaged
from lamb.dyneval import Dyneval
import numpy as np
import six
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest
FLAGS = flags.FLAGS


def _load_checkpoint(checkpoint_filename, extra_vars, trainable_only=False):
  if tf.gfile.IsDirectory(checkpoint_filename):
    checkpoint_filename = tf.train.latest_checkpoint(checkpoint_filename)
  logging.info('Loading checkpoint %s', checkpoint_filename)
  saveables = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
               tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
  if trainable_only:
    saveables = list(set(saveables) & set(tf.trainable_variables()))
  # Try to restore all saveables, if that fails try without extra_vars.
  try:
    saver = tf.train.Saver(var_list=saveables)
    saver.restore(tf.get_default_session(), checkpoint_filename)
  except (ValueError, tf.errors.NotFoundError):
    logging.info('Missing key in checkpoint. Trying old checkpoint format.')
    saver = tf.train.Saver(var_list=list(set(saveables) - set(extra_vars)))
    saver.restore(tf.get_default_session(), checkpoint_filename)


def train(tuner, data, vocab, config, experiment_dir, seed=None):
  """Main training loop.

  Args:
    tuner: .
    data: .
    vocab: .
    config: A config object (see get_config()).
    experiment_dir: Path of a directory where to log training events.
    seed: suitable for tf.set_random_seed

  Returns:
    The second return value of _maybe_report_measure.
  """

  if FLAGS.save_config:
    config.save(os.path.join(experiment_dir, 'config'))

  session_config = tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement)
  with tf.Graph().as_default():
    tf.set_random_seed(seed)
    logging.info('Creating the model.')
    config = lamb_flags.handle_config_defaults(config, lm.LM.num_params)
    model = lm.LM(config)
    logging.info('Model created.')

    if FLAGS.trigger_averaging_turns >= 0:
      averaged = Averaged(tf.trainable_variables())
    else:
      averaged = None

    # The monitor and the lr scheduler have some state that we need to
    # checkpoint in case of preemption. We do that by serializing them into the
    # graph.
    training_state = utils.TFSerializer('training_state')
    def sync_training_state_from_graph():
      state = training_state.retrieve()
      logging.info('Loaded training state: %s', state)
      if state.get('monitor_state', None):
        monitor.set_state(state['monitor_state'])
      if state.get('learning_rate_state', None):
        lr_scheduler.set_state(state['learning_rate_state'])
    def sync_training_state_to_graph():
      state = {
          # To help maintain backwards compatibility.
          'state_version': 1,
          'monitor_state': monitor.state(),
          'learning_rate_state': lr_scheduler.state()
      }
      training_state.store(state)

    # Checkpoint saving.
    logging.info('Creating savers.')
    best_turn_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    last_turn_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    best_checkpoint_dir = os.path.join(experiment_dir, 'best/')
    last_checkpoint_dir = os.path.join(experiment_dir, 'last/')
    best_checkpoint_filename = os.path.join(best_checkpoint_dir, 'model.ckpt')
    last_checkpoint_filename = os.path.join(last_checkpoint_dir, 'model.ckpt')
    # Upon resuming from a checkpoint the saver won't count the old checkpoints
    # against max_to_keep. Recover its state.
    best_checkpoint_states = tf.train.get_checkpoint_state(best_checkpoint_dir)
    last_checkpoint_states = tf.train.get_checkpoint_state(last_checkpoint_dir)
    if best_checkpoint_states is not None:
      logging.info('Previous best checkpoint paths: %s',
                   best_checkpoint_states.all_model_checkpoint_paths)
      best_turn_saver.recover_last_checkpoints(
          best_checkpoint_states.all_model_checkpoint_paths)
    if last_checkpoint_states is not None:
      logging.info('Previous last checkpoint paths: %s',
                   last_checkpoint_states.all_model_checkpoint_paths)
      last_turn_saver.recover_last_checkpoints(
          last_checkpoint_states.all_model_checkpoint_paths)
    def maybe_save_checkpoint(saver, filename):
      if FLAGS.save_checkpoints:
        logging.info('Saving checkpoint %s', filename)
        sync_training_state_to_graph()
        saver.save(tf.get_default_session(), filename,
                   global_step=model.global_step())
    # Callback for monitor.
    def save_best_checkpoint():
      maybe_save_checkpoint(best_turn_saver, best_checkpoint_filename)
    # Callback for train_loop.
    def save_last_checkpoint():
      maybe_save_checkpoint(last_turn_saver, last_checkpoint_filename)

    # The monitor keeps track of the best result so far, does early stopping.
    monitor = monitoring.TrainingMonitor(
        max_turns=config.turns,
        tuner=tuner,
        new_best_fn=save_best_checkpoint,
        es_turns=FLAGS.early_stopping_turns,
        es_rampup_turns=FLAGS.early_stopping_rampup_turns,
        es_slowest_rate=FLAGS.early_stopping_slowest_rate)
    # Set up the learning rate scheduler
    lr_scheduler = monitoring.LearningRateScheduler(
        base_learning_rate=config.learning_rate,
        monitor=monitor,
        drop_multiplier=config.drop_learning_rate_multiplier,
        drop_turns=config.drop_learning_rate_turns,
        drop_at_turn_at_the_latest=config.drop_learning_rate_at_the_latest)

    with tf.Session(config=session_config) as sess:
      logging.info('Initializing model.')
      sess.run(tf.global_variables_initializer())

      # Load the checkpoint specified by the user or try to resume from last.
      if FLAGS.load_checkpoint:
        checkpoint_filename = os.path.join(experiment_dir,
                                           FLAGS.load_checkpoint)
        _load_checkpoint(checkpoint_filename, training_state.variables(),
                         not FLAGS.load_optimizer_state)
        if FLAGS.load_optimizer_state:
          sync_training_state_from_graph()
        if averaged and FLAGS.load_averaged:
          averaged.switch_to_average()
          averaged.reset()
      else:
        try:
          _load_checkpoint(last_checkpoint_dir, training_state.variables())
          sync_training_state_from_graph()
          # TODO(melisgl): The training iterator state and last_state are not
          # saved currently. They should be, of course, but failing that random
          # initialization of dataset iterators ensures that there is no bias
          # introduced if training is repeatedly interrupted and continued from
          # a checkpoint. So use a random seed in this case.
          random.seed()
          np.random.seed()
        except (ValueError, tf.errors.NotFoundError):
          logging.info('Last checkpoint file %s does not exist.',
                       last_checkpoint_filename)

      # Takes a lot of space. Disabled for now.
      # summary_writer = tf.summary.FileWriter(
      #     experiment_dir, graph=sess.graph,
      #     flush_secs=FLAGS.summary_flush_secs)
      summary_writer = None

      if FLAGS.dyneval:
        dyneval = Dyneval(model.clipped_grads_and_vars,
                          learning_rate=FLAGS.dyneval_learning_rate,
                          decay_rate=FLAGS.dyneval_decay_rate,
                          epsilon=FLAGS.dyneval_epsilon)
      else:
        dyneval = None

      if config.turns > 0:
        logging.info('Starting training.')
      else:
        logging.info('Starting testing.')
      metrics = _train_loop(
          monitor, lr_scheduler, averaged, dyneval, model, data, vocab, config,
          summary_writer, save_last_checkpoint)
      logging.info('Training finished.')

      return metrics, monitor.turn()


def _train_loop(monitor, lr_scheduler, averaged, dyneval, model,
                data, vocab, config, summary_writer, save_last_checkpoint_fn):
  source_iterator = corpus.get_batches(
      data['training'], vocab,
      config.batch_size,
      config.max_time_steps,
      num_samples=config.num_training_samples,
      episodic=FLAGS.episodic,
      deterministic=False,
      conditioning_separator=config.conditioning_separator)
  last_state = None
  steps_per_sec = 0.0

  def munge_max_batches_flag_value(max_batches):
    if max_batches == -1:
      return None
    else:
      return max_batches

  def evaluate0():
    # KLUDGE: This depends on monitor calling this function before using the
    # worst target.
    monitor.set_es_worst_target(es_worst_target())
    global_step = model.global_step()
    logging.info('turn: %s (eval), step: %d (opt) (%.2f/s)',
                 monitor.turn(), global_step, steps_per_sec)
    if config.accum_batch_size == -1:
      eval_batch_size = config.batch_size
    else:
      eval_batch_size = config.accum_batch_size
    training_xe, valid_xe, test_xe = evaluation.evaluate_all(
        model, data, vocab, eval_batch_size, config.max_time_steps,
        FLAGS.min_non_episodic_eval_examples_per_stripe,
        munge_max_batches_flag_value(FLAGS.max_training_eval_batches),
        munge_max_batches_flag_value(FLAGS.max_eval_eval_batches),
        munge_max_batches_flag_value(FLAGS.max_test_eval_batches),
        FLAGS.episodic,
        config.eval_softmax_temperature,
        config.eval_softmax_temperature_estimation_num_tokens,
        config.eval_method,
        config.num_eval_samples,
        config.eval_power_mean_power,
        config.eval_dropout_multiplier,
        config.validation_prediction_file,
        dyneval,
        conditioning_separator=config.conditioning_separator)
    return valid_xe, {'training_xe': training_xe,
                      'test_xe': test_xe,
                      'global_step': global_step}

  def evaluate():
    if monitor.averaging_triggered():
      with averaged:
        logging.info('Evaluating with averaged parameters.')
        return evaluate0()
    else:
      return evaluate0()

  def add_summary(summary_str):
    if summary_writer is not None:
      summary_writer.add_summary(summary_str, model.global_step())

  def add_summaries_for_metrics():
    metrics = monitor.metrics()
    summary = tf.Summary()
    for key in metrics:
      summary.value.add(tag=key, simple_value=metrics[key])
    add_summary(summary)

  # Compute the early stopping worst target. It may change when the learning
  # rate is dropped.
  def es_worst_target():
    if FLAGS.early_stopping_worst_xe_target is None:
      return -1.0
    else:
      targets_for_lr_drops = [
          float(string) for string
          in FLAGS.early_stopping_worst_xe_target.split(',')
          if string
      ]
      num_drops = lr_scheduler.num_drops()
      if targets_for_lr_drops:
        return targets_for_lr_drops[min(num_drops, len(targets_for_lr_drops)-1)]
      else:
        return None

  def log_summaries(summary):
    utils.log_scalar_summaries(summary)
    add_summary(summary)

  while monitor.next_turn(evaluate):

    logging.info('metrics: %r', monitor.metrics())
    logging.info(
        'early stopping: turns: %s, worst xe target: %s, best expected xe: %s',
        monitor.effective_es_turns(), monitor.es_worst_target(),
        monitor.best_expected_xe())
    add_summaries_for_metrics()

    # If enough turns passed without improvement, turn on averaging.
    best_turn = monitor.best_xe_turn() or 0
    num_tuns_since_best = monitor.turn() - best_turn
    if (averaged and
        ((monitor.turn() > 0 and
          num_tuns_since_best >= FLAGS.trigger_averaging_turns) or
         (FLAGS.trigger_averaging_at_the_latest >= 0 and
          monitor.turn() >= FLAGS.trigger_averaging_at_the_latest))):
      monitor.set_averaging_triggered(True)

    start_time = time.time()
    sum_cost = 0.0
    sum_tokens = 0
    for _ in range(FLAGS.steps_per_turn):
      cost, summary, last_state, num_tokens = train_1(
          model, source_iterator, last_state,
          learning_rate=lr_scheduler.learning_rate(),
          accum_batch_size=model.config.accum_batch_size)
      if monitor.averaging_triggered():
        averaged.take_sample()
      sum_cost += cost
      sum_tokens += num_tokens
      # Log summaries at the very beginning of training to make it easier to
      # debug initialization problems.
      if (model.global_step() == 1 or
          (model.global_step()+1) %
          FLAGS.print_training_stats_every_num_steps == 1):
        log_summaries(summary)
        logging.info('avg training cost at step %d: %.5f',
                     model.global_step(), sum_cost / sum_tokens)
        sum_cost = 0.0
        sum_tokens = 0
    steps_per_sec = FLAGS.steps_per_turn / (time.time()-start_time)

    # TODO(melisgl): Is this the right frequency for saving?
    save_last_checkpoint_fn()

  metrics = monitor.metrics()
  logging.info('Finished at turn %d for reason: %s',
               monitor.turn(), monitor.finished_reason())
  logging.info('Best XE was %5.5f at turn %d',
               metrics['best_xe'], metrics['best_xe_turn'])
  return metrics


def train_1(model, source_iterator, last_state,
            learning_rate, extra_feed=None, accum_batch_size=-1):
  """Trains model for a a single iteration."""
  if accum_batch_size == -1:
    cond, cond_len, source, source_len, target = next(source_iterator)
    feed = _make_train_feed(model, cond, cond_len, source, source_len, target,
                            last_state, learning_rate, extra_feed)
    batch_size = feed[model.source_len].shape[0]
    num_tokens = feed[model.source_len].sum()
    cost, summary, last_state = model.fit(feed)
    return cost*batch_size, summary, last_state, num_tokens
  else:
    return _train_1_with_accum(model, source_iterator, last_state,
                               learning_rate, extra_feed, accum_batch_size)


def _train_1_with_accum(model, source_iterator, last_state,
                        learning_rate, extra_feed, accum_batch_size):
  """Trains model for a a single iteration."""
  cond, cond_len, source, source_len, target = next(source_iterator)
  (conds, cond_lens, sources, source_lens,
   targets, last_states) = _maybe_split_batch(
       cond, cond_len, source, source_len, target, last_state, accum_batch_size)
  num_accum_batches = len(sources)
  cost = 0.0
  new_last_states = []
  batch_size = 0
  num_tokens = 0
  for i in six.moves.range(num_accum_batches):
    cond = conds[i] if cond is not None else None
    cond_len = cond_lens[i] if cond_len is not None else None
    source = sources[i]
    source_len = source_lens[i]
    target = targets[i]
    if last_states is not None:
      last_state = last_states[i]
    else:
      last_state = None
    feed = _make_train_feed(model, cond, cond_len, source, source_len, target,
                            last_state, learning_rate, extra_feed)
    batch_size1 = feed[model.source_len].shape[0]
    batch_size += batch_size1
    num_tokens += feed[model.source_len].sum()
    cost1, summary1, last_state1 = model.accumulate_gradients(feed)
    cost += cost1*batch_size1
    new_last_states.append(last_state1)
  model.fit_accumulated(feed)
  last_state = _concat_last_states(new_last_states)
  return cost, summary1, last_state, num_tokens


def _make_train_feed(model, cond, cond_len, source, source_len, target,
                     last_state, learning_rate, extra_feed=None):
  feed = {}
  model.add_input_to_feed(feed, cond, cond_len, source, source_len, target)
  model.add_dropout_to_feed(feed)
  feed.update({
      model.num_samples: model.config.num_training_samples,
      model.learning_rate: learning_rate
  })
  if extra_feed:
    feed.update(extra_feed)
  if not FLAGS.episodic and last_state is not None:
    # At test time we start from zero state, so let's forget the
    # current state during training too. Simply not feeding the
    # previous state back would be simpler, but it distorts the
    # objective too much.
    if model.config.drop_state_probability > 0.0:
      mask = [None]
      def ensure_mask(x):
        if mask[0] is None:
          mask[0] = np.random.binomial(
              1, 1.0-model.config.drop_state_probability,
              size=[x.shape[0]*model.config.num_training_samples, 1])
        return mask[0]
      last_state = utils.map_nested(lambda x: ensure_mask(x)*x, last_state)
    feed.update({model.initial_state: last_state})
  return feed


def _maybe_split_batch(cond, cond_len, source, source_len, target, last_state,
                       accum_batch_size):
  batch_size = source_len.shape[0]
  assert batch_size % accum_batch_size == 0
  n = batch_size // accum_batch_size
  return (np.split(cond, n, axis=1) if cond is not None else None,
          np.split(cond_len, n, axis=0) if cond_len is not None else None,
          np.split(source, n, axis=1),
          np.split(source_len, n, axis=0),
          np.split(target, n, axis=1),
          _split_last_state(last_state, n) if last_state is not None else None)


def _split_last_state(last_state, n):
  list_of_split_arrays = [np.split(array, n)
                          for array in nest.flatten(last_state)]
  list_of_split_states = zip(*list_of_split_arrays)
  return [nest.pack_sequence_as(last_state, split_state)
          for split_state in list_of_split_states]


def _concat_last_states(last_states):
  list_of_flat_states = [nest.flatten(last_state) for last_state in last_states]
  flat_list_of_states = zip(*list_of_flat_states)
  flat_state = [np.concatenate(list_of_states, axis=0) for list_of_states
                in flat_list_of_states]
  return nest.pack_sequence_as(last_states[0], flat_state)
