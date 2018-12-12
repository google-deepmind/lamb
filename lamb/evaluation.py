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

"""Deterministic and MC evaluation."""

# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import logging
from lamb import corpus
import numpy as np
import six
import tensorflow.compat.v1 as tf


def _make_feed(model, cond, cond_len, source, source_len, target,
               last_state, episodic, dropout_multiplier=1.0, temperature=1.0):
  feed = {model.softmax_temperature: temperature}
  model.add_input_to_feed(feed, cond, cond_len, source, source_len, target)
  model.add_dropout_to_feed(feed, dropout_multiplier)
  if not episodic and last_state is not None:
    feed.update({model.initial_state: last_state})
  return feed


def _sum_masked(source_len, xe):
  mask = (np.arange(xe.shape[0]).reshape(1, -1) <
          source_len.reshape(-1, 1)).astype(np.float32).transpose()
  return np.sum(mask*xe), np.sum(mask)


def evaluate_deterministic(
    model, make_data_iterator_fn, dataset_name, episodic,
    num_batches_to_discard=0, temperature=1.0, print_level=2,
    prediction_callback=None, extra_ops=()):
  """Evaluate with a single pass with dropout turned off."""
  sum_xe = 0
  sum_len = 0
  num_batches = 0
  last_state = None
  for (cond, cond_len, source, source_len, target) in make_data_iterator_fn():
    feed = _make_feed(model, cond, cond_len, source, source_len, target,
                      last_state, episodic, 0, temperature)
    xe, last_state = tf.get_default_session().run(
        [model.xe_losses, model.last_state]+list(extra_ops), feed)[0:2]
    if num_batches >= num_batches_to_discard:
      sum_xe1, sum_len1 = _sum_masked(source_len, xe)
      sum_xe += sum_xe1
      sum_len += sum_len1
    if prediction_callback:
      prediction_callback(target, source_len, xe)
    num_batches += 1
  average_xe = sum_xe / sum_len
  if print_level >= 1:
    logging.info('final %s xe: %6.5f (%s), batches: %s',
                 dataset_name, average_xe, sum_len,
                 num_batches-num_batches_to_discard)
  return average_xe


def evaluate_mc_with_geometric_mean(
    model, make_data_iterator_fn, dataset_name, episodic,
    num_batches_to_discard=0, num_samples=0, dropout_multiplier=1.0,
    temperature=1.0, print_level=2):
  """Evaluate with MC dropout, taking the geometric mean of predictions."""
  assert num_samples > 0
  # Loop over batches made of examples in the corpus and accumulate
  # statistics. data_iterator is already set up properly for
  # episodic or non-episodic mode, we just need to keep passing the
  # network's last state back.
  sum_xe = 0
  sum_len = 0
  num_batches = 0
  last_state = None
  for (cond, cond_len, source, source_len, target) in make_data_iterator_fn():
    if num_batches >= num_batches_to_discard:
      feed = _make_feed(model, cond, cond_len, source, source_len, target,
                        last_state, episodic, dropout_multiplier, 1.0)
      sum_logits = None
      for _ in six.moves.range(num_samples):
        # TODO(melisgl): The transfer of the logits
        # [output_embedding_size, vocab_size] is expensive. The
        # summing should take place purely in the graph.
        logits = tf.get_default_session().run(model.logits, feed)
        if sum_logits is None:
          sum_logits = logits
        else:
          sum_logits += logits
      # Now we feed the average of per-sample logits through the
      # softmax to normalize.
      feed.update({model.logits: sum_logits/num_samples,
                   model.softmax_temperature: temperature})
      xe = tf.get_default_session().run(model.xe_losses, feed)
      sum_xe1, sum_len1 = _sum_masked(source_len, xe)
      sum_xe += sum_xe1
      sum_len += sum_len1
      average_xe = sum_xe / sum_len
      if print_level >= 2:
        logging.info('%s xe: %6.5f (%s), batches: %s (ns=%s)',
                     dataset_name, average_xe, sum_len,
                     num_batches-num_batches_to_discard, num_samples)
    # Do a deterministic pass. We need this even in MC mode to get a
    # single last state to feed back for the next batch. This isn't
    # strictly correct, we should actually sample the state to be
    # fed back and the most obvious way to do that is to loop over
    # the entire dataset instead of just the batch, but then we'd
    # need to keep all those logits around.
    feed = _make_feed(model, cond, cond_len, source, source_len, target,
                      last_state, episodic, 0, temperature)
    last_state = tf.get_default_session().run(model.last_state, feed)
    num_batches += 1
  if print_level >= 1:
    logging.info('final %s xe: %6.5f (%s), batches: %s (ns=%s)',
                 dataset_name, average_xe, sum_len,
                 num_batches-num_batches_to_discard, num_samples)
  return average_xe


def evaluate_mc_with_power_mean(
    model, make_data_iterator_fn, dataset_name, episodic,
    num_batches_to_discard=0, num_samples=0, dropout_multiplier=1.0,
    power=1.0, temperature=1.0, print_level=2):
  # pylint: disable=g-doc-return-or-yield,g-doc-args
  # pylint: disable=g-docstring-missing-newline
  r"""Evaluate with MC dropout, taking the power mean of predictions.

  M_p(x_1,...,x_n) = (1/n*\sum_{i=1}^n x_i^p)^{1/p}"""
  assert num_samples > 0
  # Loop over batches made of examples in the corpus and accumulate
  # statistics. data_iterator is already set up properly for
  # episodic or non-episodic mode, we just need to keep passing the
  # network's last state back.
  sum_xe = 0
  sum_len = 0
  num_batches = 0
  last_state = None
  for (cond, cond_len, source, source_len, target) in make_data_iterator_fn():
    if num_batches >= num_batches_to_discard:
      feed = _make_feed(model, cond, cond_len, source, source_len, target,
                        last_state, episodic, dropout_multiplier, 1.0)
      log_sum_probs = None
      for _ in six.moves.range(num_samples):
        # TODO(melisgl): The transfer of the logits
        # [output_embedding_size, vocab_size] is expensive. The
        # summing should take place purely in the graph.
        log_probs = tf.get_default_session().run(model.log_probs, feed)
        if log_sum_probs is None:
          log_sum_probs = log_probs
        else:
          log_sum_probs = np.logaddexp(log_sum_probs, power*log_probs)
      # log_sum_probs is \ln \sum x_i^p. We need to divide by n
      # (i.e. subtract math.log(num_samples)), and raise to the
      # power 1/p (i.e. divide by power), then feed the results
      # through the softmax to renormalize.
      feed.update(
          {model.logits: (log_sum_probs -
                          math.log(num_samples))/power*temperature})
      xe = tf.get_default_session().run(model.xe_losses, feed)
      sum_xe1, sum_len1 = _sum_masked(source_len, xe)
      sum_xe += sum_xe1
      sum_len += sum_len1
      average_xe = sum_xe / sum_len
      if print_level >= 2:
        logging.info('%s xe: %6.5f (%s), batches: %s (ns=%s)',
                     dataset_name, average_xe, sum_len,
                     num_batches-num_batches_to_discard, num_samples)
    # Do a deterministic pass. We need this even in MC mode to get a
    # single last state to feed back for the next batch. This isn't
    # strictly correct, we should actually sample the state to be
    # fed back and the most obvious way to do that is to loop over
    # the entire dataset instead of just the batch, but then we'd
    # need to keep all those logits around.
    feed = _make_feed(model, cond, cond_len, source, source_len, target,
                      last_state, episodic, 0, 1.0)
    last_state = tf.get_default_session().run(model.last_state, feed)
    num_batches += 1
  if print_level >= 1:
    logging.info('final %s xe: %6.5f (%s), batches: %s (ns=%s)',
                 dataset_name, average_xe, sum_len,
                 num_batches-num_batches_to_discard, num_samples)
  return average_xe


def evaluate_mc_with_arithmetic_mean(
    model, make_data_iterator_fn, dataset_name, episodic,
    num_batches_to_discard=0, num_samples=0, dropout_multiplier=1.0,
    temperature=1.0, deterministic_last_state=False, print_level=2,
    dyneval=None, extra_ops=()):
  """Evaluate with MC dropout, taking the average of predictions."""
  assert num_samples > 0
  log_sum_prob = None
  for sample_index in six.moves.range(num_samples):
    num_batches = 0
    start = 0
    sum_xe = 0.0
    sum_len = 0
    last_state = None
    if dyneval:
      # Restore the original weights.
      dyneval.restore()
    for (cond, cond_len, source, source_len, target) in make_data_iterator_fn():
      feed = _make_feed(model, cond, cond_len, source, source_len, target,
                        last_state, episodic, dropout_multiplier, temperature)
      # [time, batch]
      sample_xe, last_state = tf.get_default_session().run(
          [model.xe_losses, model.last_state]+list(extra_ops), feed)[0:2]
      if num_batches >= num_batches_to_discard:
        # [batch, time]
        log_probs = -np.transpose(sample_xe).astype(np.float64)
        max_time_steps = log_probs.shape[1]
        if log_sum_prob is None:
          log_sum_prob = log_probs
        elif sample_index == 0:
          log_sum_prob = np.concatenate([log_sum_prob, log_probs], axis=1)
        else:
          log_sum_prob[:, start:start+max_time_steps] = np.logaddexp(
              log_sum_prob[:, start:start+max_time_steps], log_probs)
        # Compute the mean XE.
        mask = (np.arange(max_time_steps).reshape(1, -1) <
                source_len.reshape(-1, 1)).astype(np.float32)
        sum_xe += -np.sum(mask*(log_sum_prob[:, start:start+max_time_steps] -
                                math.log(sample_index+1)))
        sum_len += np.sum(mask)
        start += max_time_steps
        if deterministic_last_state:
          feed = _make_feed(model, cond, cond_len, source, source_len, target,
                            last_state, episodic, 0.0, temperature)
          last_state = tf.get_default_session().run(model.last_state, feed)
      num_batches += 1
    average_xe = sum_xe / sum_len
    if print_level >= 2:
      logging.info('%s xe: %6.5f (%s), batches: %s (ns=%s/%s)',
                   dataset_name, average_xe, sum_len,
                   num_batches-num_batches_to_discard, sample_index+1,
                   num_samples)
  if print_level >= 1:
    logging.info('final %s xe: %6.5f (%s), batches: %s (ns=%s)',
                 dataset_name, average_xe, sum_len,
                 num_batches-num_batches_to_discard, num_samples)
  return average_xe


def evaluate(model, make_data_iterator_fn, vocab, dataset_name, episodic,
             num_batches_to_discard=0, num_samples=0, dropout_multiplier=1.0,
             eval_method=None, power=1.0, temperature=1.0, print_level=2,
             prediction_file=None, dyneval=None, extra_ops=()):
  """Evaluate."""
  vocab = vocab
  name = dataset_name

  assert eval_method in ['deterministic', 'geometric', 'power', 'arithmetic']

  if eval_method == 'deterministic':
    name += '_det'
  else:
    if eval_method == 'geometric':
      name += '_mcg'
    elif eval_method == 'power':
      name += '_mcp' + str(power)
    elif eval_method == 'arithmetic':
      name += '_mca'
    if dropout_multiplier != 1.0:
      name += '_d' + str(dropout_multiplier)

  if temperature != 1.0:
    name += '_t' + str(temperature)

  def dispatch0(callback, extra_ops):
    if eval_method == 'deterministic':
      return evaluate_deterministic(
          model, make_data_iterator_fn, name, episodic, num_batches_to_discard,
          temperature, print_level=print_level, prediction_callback=callback,
          extra_ops=extra_ops)
    elif eval_method == 'geometric':
      assert not extra_ops, 'Not implemented.'
      return evaluate_mc_with_geometric_mean(
          model, make_data_iterator_fn, name, episodic, num_batches_to_discard,
          num_samples, dropout_multiplier, temperature,
          print_level=print_level)
    elif eval_method == 'power':
      assert not extra_ops, 'Not implemented.'
      return evaluate_mc_with_power_mean(
          model, make_data_iterator_fn, name, episodic, num_batches_to_discard,
          num_samples, dropout_multiplier, power, temperature,
          print_level=print_level)
    elif eval_method == 'arithmetic':
      return evaluate_mc_with_arithmetic_mean(
          model, make_data_iterator_fn, name, episodic, num_batches_to_discard,
          num_samples, dropout_multiplier, temperature,
          print_level=print_level, dyneval=dyneval,
          extra_ops=extra_ops)
    else:
      assert False

  def dispatch(callback):
    if dyneval is None:
      return dispatch0(callback, list(extra_ops))
    else:
      with dyneval:
        return dispatch0(callback, [dyneval.update_op()] + list(extra_ops))

  if prediction_file:
    with tf.gfile.GFile(prediction_file, 'w') as f:
      def callback(target, length, xe):
        for i in six.moves.range(length[0]):
          token = vocab.decode([target[i][0]])[0]
          log_prob = -xe[i][0]
          print('{!r}'.format(token), file=f)
          print('{}'.format(log_prob), file=f)
      return dispatch(callback)
  else:
    return dispatch(None)


def evaluate_all(model, data, vocab, batch_size, max_time_steps,
                 min_non_episodic_eval_examples_per_stripe,
                 max_training_eval_batches,
                 max_eval_eval_batches,
                 max_test_eval_batches,
                 episodic,
                 eval_softmax_temperature,
                 eval_softmax_temperature_estimation_num_tokens,
                 eval_method,
                 num_eval_samples,
                 eval_power_mean_power,
                 eval_dropout_multiplier,
                 validation_prediction_file,
                 dyneval,
                 conditioning_separator=None):
  """Evaluate on training/validation and maybe on test."""
  # Evaluate on training set.
  def make_train_iterator():
    return corpus.get_batches(
        data['training'], vocab,
        batch_size,
        max_time_steps,
        episodic=episodic,
        deterministic=True,
        max_epochs=1,
        max_batches=max_training_eval_batches,
        conditioning_separator=conditioning_separator)

  if dyneval:
    dyneval.zero_sum_squared_grads()
    training_xe = evaluate(
        model, make_train_iterator, vocab, 'train', episodic,
        # Let's not waste time on samples on training evaluation.
        num_samples=0, eval_method='deterministic',
        extra_ops=[dyneval.add_squared_grads_op()])
  else:
    training_xe = evaluate(
        model, make_train_iterator, vocab, 'train', episodic,
        # Let's not waste time on samples on training evaluation.
        num_samples=0, eval_method='deterministic')

  # For expediency, we evaluate in batches even in non-episodic
  # mode, but make sure that each stripe in the batch has at least
  # min_non_episodic_eval_examples_per_stripe examples to limit
  # the handicap.
  def eval_batch_size(dataset):
    if (not episodic and
        min_non_episodic_eval_examples_per_stripe):
      return max(1, min(batch_size, dataset.size() //
                        min_non_episodic_eval_examples_per_stripe))
    else:
      return batch_size

  # Find the best softmax temperature on using a few batches of the
  # validation set.
  config_sttt = eval_softmax_temperature
  if config_sttt > 0.0:
    eval_softmax_temperature = config_sttt
  else:
    def make_quick_eval_iterator():
      quick_eval_max_batches = max(
          1,
          eval_softmax_temperature_estimation_num_tokens //
          batch_size // max_time_steps)
      return corpus.get_batches(
          data['valid'], vocab,
          batch_size,
          max_time_steps,
          episodic=episodic,
          deterministic=True,
          max_epochs=1,
          max_batches=quick_eval_max_batches,
          conditioning_separator=conditioning_separator)
    xe = 99999999.0
    best_i = 0
    for i in six.moves.range(50):
      eval_softmax_temperature0 = 1.06-i*0.02
      if eval_softmax_temperature0 < -config_sttt-0.0001:
        break
      xe0 = evaluate(
          model, make_quick_eval_iterator, vocab, 'sttt_eval', episodic,
          num_samples=num_eval_samples,
          temperature=eval_softmax_temperature0,
          power=eval_power_mean_power,
          dropout_multiplier=eval_dropout_multiplier,
          eval_method=eval_method,
          print_level=1)
      if xe0 < xe:
        best_i = i
        xe = xe0
        eval_softmax_temperature = eval_softmax_temperature0
      # Stop if there was no improvement for two rounds.
      if best_i+1 < i:
        break

  # Use the best eval_softmax_temperature and do a longer
  # run on the validation set.
  def make_eval_iterator():
    return corpus.get_batches(
        data['valid'], vocab,
        eval_batch_size(data['valid']) if not dyneval else 1,
        max_time_steps,
        episodic=episodic,
        deterministic=True,
        max_epochs=1,
        max_batches=max_eval_eval_batches,
        conditioning_separator=conditioning_separator)
  xe = evaluate(
      model, make_eval_iterator, vocab, 'valid', episodic,
      num_samples=num_eval_samples,
      temperature=eval_softmax_temperature,
      power=eval_power_mean_power,
      dropout_multiplier=eval_dropout_multiplier,
      eval_method=eval_method,
      prediction_file=validation_prediction_file,
      dyneval=dyneval)

  # evaluate on test
  if data['test'].size():
    def make_test_iterator():
      return corpus.get_batches(
          data['test'], vocab,
          eval_batch_size(data['test']) if not dyneval else 1,
          max_time_steps,
          episodic=episodic,
          deterministic=True,
          max_epochs=1,
          max_batches=max_test_eval_batches,
          conditioning_separator=conditioning_separator)
    test_xe = evaluate(
        model, make_test_iterator, vocab, 'test', episodic,
        num_samples=num_eval_samples,
        temperature=eval_softmax_temperature,
        power=eval_power_mean_power,
        dropout_multiplier=eval_dropout_multiplier,
        eval_method=eval_method,
        dyneval=dyneval)
  else:
    test_xe = None

  return training_xe, xe, test_xe
