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

"""Runner for the lamb model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import traceback

from absl import app
from absl import flags
from absl import logging
from lamb import corpus
from lamb import lamb_flags
from lamb import training
from lamb import utils
from lamb.vocab import Vocab
import numpy as np
import six
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS


class Experiment(object):
  """Experiment class."""

  def __init__(self, config, experiment_dir, tuner):
    self._experiment_dir = experiment_dir
    self._tuner = tuner
    self._config = config
    self._training_state = None

  def _finalize_config(self, config, data, vocab):
    """Apply data dependent defaults to config."""
    config.vocab_size = vocab.size()
    # TODO(melisgl): Separate the vocabs.
    config.conditioning_vocab_size = vocab.size()
    config.eos_index = vocab.eos_index()

    # Set up max_time_steps.
    if FLAGS.episodic:
      # Calculate maximum number of time steps. Add 1 for the end token.
      config.max_time_steps = min(
          config.max_time_steps,
          max([data['training'].max_sentence_length(),
               data['valid'].max_sentence_length(),
               data['test'].max_sentence_length()]) + 1)

    return config

  def average_metrics(self, metrics_list):
    n = len(metrics_list)
    if n > 0:
      average = {}
      for key in ['best_xe']:
        average[key] = sum([metrics[key] for metrics in metrics_list]) / n
      return average
    else:
      return None

  def valid_metrics(self, metrics):
    for value in metrics.values():
      if math.isnan(value):
        return False
    return True

  def final_measure(self, fold_metrics):
    average = self.average_metrics(fold_metrics)
    if not self.valid_metrics(average):
      return None
    else:
      return average['best_xe']

  def run_training(self, folds):
    if self._tuner:
      self._run_training_with_tuner(folds)
    else:
      self._run_training_without_tuner(folds)

  def _run_training_without_tuner(self, folds):
    """Simply train. Don't modify the configuration settings."""
    tf.gfile.MakeDirs(self._experiment_dir)
    fold_metrics = []
    for i, (data, vocab) in enumerate(folds):
      logging.info('Training on fold %d/%d', i+1, len(folds))

      config = self._finalize_config(self._config, data, vocab)
      metrics, _ = training.train(None, data, vocab, config,
                                  self._experiment_dir, seed=FLAGS.seed + i)
      logging.info('Training on fold %d/%d measure: %s', i+1, len(folds),
                   metrics)
      fold_metrics.append(metrics)

    average_metrics = self.average_metrics(fold_metrics)
    logging.info('Average after %d folds: %s', len(fold_metrics),
                 average_metrics)
    logging.info('Crossvalidation results:')
    for i, metrics in enumerate(fold_metrics):
      logging.info('Fold %i: %s', i, metrics)

  def _run_training_with_tuner(self, folds):
    """Train and evaluate based on parameters provided from a tuner."""
    try:
      fold_metrics = []
      sum_turns = 0
      for i, (data, vocab) in enumerate(folds):
        logging.info('Training on fold %d/%d', i+1, len(folds))

        config = self._finalize_config(self._config, data, vocab)

        # Setup the experiment directory.
        exp_actual_dir = self._experiment_dir
        if len(folds) > 1:
          exp_actual_dir = os.path.join(exp_actual_dir,
                                        '_fold{}'.format(i),
                                        '')
        tf.gfile.MakeDirs(exp_actual_dir)

        # Train.
        metrics, turn = training.train(
            self._tuner, data, vocab, config,
            exp_actual_dir, seed=FLAGS.seed + i)
        logging.info('Training on fold %d/%d metrics: %s', i+1, len(folds),
                     metrics)
        fold_metrics.append(metrics)
        sum_turns += turn

        # Report average measure across folds up to now to the tuner.
        average_metrics = self.average_metrics(fold_metrics)
        logging.info('Average after %d folds: %s', len(fold_metrics),
                     average_metrics)
        measure = self.final_measure(fold_metrics)
        if measure is None:
          self._tuner.report_done(infeasible=True, infeasible_reason='nan')
          return
        if self._tuner.report_measure(measure, global_step=sum_turns+1,
                                      metrics=average_metrics):
          logging.info('Stopping due to tuner request.')
          break
    except (tf.errors.ResourceExhaustedError,
            # Some OOM conditions turn into internal errors.
            tf.errors.InternalError,
            # Ignore NaNs detected by clip_by_global_norm.
            tf.errors.InvalidArgumentError):
      stack_trace = traceback.format_exc()
      logging.warning('Reporting trial infeasible because of:\n%s', stack_trace)
      self._tuner.report_done(infeasible=True, infeasible_reason=stack_trace)
      return

    logging.info('Crossvalidation results:')
    for i, metrics in enumerate(fold_metrics):
      logging.info('Fold %i: %s', i, metrics)
    self._tuner.report_done()


def _make_fold(training_corpus, valid_corpus, test_corpus):
  """Create a data, vocab pair."""
  data = {
      'training': training_corpus,
      'valid': valid_corpus,
      'test': test_corpus,
  }
  vocab = Vocab(data['training'].tokens())
  return data, vocab


def read_corpus(filename):
  if FLAGS.word_based:
    return corpus.read_word_based_corpus(
        filename, encoding=FLAGS.file_encoding)
  else:
    return corpus.read_character_based_corpus(
        filename, encoding=FLAGS.file_encoding)


def main(argv, tuner=None):
  """Main function."""

  assert argv is None or len(argv) == 1, (
      'This program expects no non-option arguments. Got {}.'.format(argv))

  tf.enable_resource_variables()
  lamb_flags.initialize()

  if FLAGS.use_old_linear_names:
    utils._BIAS_VARIABLE_NAME = 'biases'  # pylint: disable=protected-access
    utils._WEIGHTS_VARIABLE_NAME = 'weights'  # pylint: disable=protected-access

  # Set seeds. The tensorflow seed is set later.
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  # Load the files.
  assert FLAGS.training_file, 'No training file specified.'
  training_file_data = read_corpus(FLAGS.training_file)
  if FLAGS.test_file and FLAGS.eval_on_test:
    test_file_data = read_corpus(FLAGS.test_file)
  else:
    test_file_data = corpus.Corpus(data=[])

  # Let's assemble the 'folds': training and eval set combinations,
  # plus the vocabulary.
  folds = []
  def add_fold(training_corpus, eval_corpus, test_corpus):
    fold = _make_fold(training_corpus, eval_corpus, test_corpus)
    logging.info('number of examples in fold %d', len(folds))
    logging.info('  training: %d', fold[0]['training'].size())
    logging.info('  valid: %d', fold[0]['valid'].size())
    logging.info('  test: %d', fold[0]['test'].size())
    folds.append(fold)

  if FLAGS.crossvalidate:
    logging.info('Doing cross-validation.')
    assert FLAGS.validation_file == ''  # pylint: disable=g-explicit-bool-comparison
    for _ in six.moves.range(FLAGS.crossvalidation_rounds):
      for training_set, validation_set in utils.cv_splits(
          training_file_data.data(), FLAGS.crossvalidation_folds):
        add_fold(corpus.Corpus(data=training_set),
                 corpus.Corpus(data=validation_set),
                 test_file_data)
  else:
    logging.info('Using dedicated eval data.')
    assert FLAGS.validation_file, 'No eval file specified.'
    validation_file_data = read_corpus(FLAGS.validation_file)
    add_fold(training_file_data, validation_file_data, test_file_data)

  experiment = Experiment(lamb_flags.get_config(), FLAGS.experiment_dir, tuner)
  experiment.run_training(folds=folds)


if __name__ == '__main__':
  app.run(main)
