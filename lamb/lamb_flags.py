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

"""Configuration and command-line flags.

The configuration is currently a flat namespace mapping options to values.
Typically the experiment shell scripts set these options and they are passed to
python as command-line arguments. See README.md for documentation of
configuration options.
"""

# pylint: disable=g-importing-member
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy
import math

from absl import flags
from absl import logging
import six
import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from tensorflow.contrib import training as contrib_training
from tensorflow.contrib.training.python.training import hparam_pb2

# Bump this on incompatible changes such as renaming an option and update
# maybe_upgrade_args_line below.
_config_version = 5


# The format of options is `(name, type, default_value, visibility)`.
# `visibility` is optional and can be `deprecated`, `external` or `internal`.
def option_visibility(option):
  if len(option) == 4:
    return option[3]
  else:
    return None


# There will be a command-line flag for every option except those with
# `internal` visibility. Conversely, in the Config object, `external` and
# `deprecated` are not going to be present. String like 'data' are turned into
# python comments when options are saved/printed.
_config_options = [
    ('config_version', 'integer', _config_version),

    'data',
    ('training_file', 'string', ''),
    ('validation_file', 'string', ''),
    ('test_file', 'string', ''),
    ('conditioning_separator', 'string', ''),
    ('file_encoding', 'string', 'utf-8'),
    ('word_based', 'boolean', False),
    ('episodic', 'boolean', False),

    'model',
    ('num_params', 'integer', -1),
    ('share_input_and_output_embeddings', 'boolean', False),
    ('input_embedding_size', 'integer', -1),
    ('output_embedding_size', 'integer', -1),
    ('input_embedding_ratio', 'float', 1.0),
    ('output_embedding_ratio', 'float', -1.0),
    ('mos_num_components', 'integer', 0),
    ('token_dropout', 'float', 0.0),
    ('embedding_dropout', 'float', 0.0),
    ('input_dropout', 'float', 0.0),
    ('output_dropout', 'float', 0.0),
    ('downprojected_output_dropout', 'float', -1.0),
    ('shared_mask_dropout', 'boolean', False),
    # Whether to embed 'globally' or per time step. They are
    # equivalent, but may differ in performance.
    ('embed_once', 'boolean', True),
    ('output_once', 'boolean', True),

    'cell',
    ('model', 'string', 'lstm'),
    ('num_layers', 'integer', 1),
    ('residual_connections', 'boolean', False),
    ('lstm_skip_connection', 'boolean', True),
    ('feature_mask_rounds', 'integer', 0),
    ('feature_mask_rank', 'integer', 0),
    # Deprecated. This is here to be able to load old configs. True sets
    # feature_mask_rounds to 2 and feature_mask_rank to 0.
    ('feature_mask', 'boolean', False),
    # If in [0,1) then within the recurrent cell in every dense
    # connectivity matrix of N elements, randomly chosen elements
    # are fixed to 0 such that the total number of trainable,
    # non-fixed values is N*sparsity_ratio. Values outside [0,1) are
    # treated as 1.0 (i.e. no sparsity),.
    ('sparsity_ratio', 'float', -1.0),
    # TODO(melisgl): Document it once it's actually used.
    ('overlay_rank', 'integer', -1),
    ('hidden_size', 'list_of_ints', '-1'),
    ('hidden_size_multiplier', 'float', 1.0),
    ('layer_norm', 'boolean', False),
    ('activation_fn', 'string', 'tf.tanh'),
    ('tie_forget_and_input_gates', 'boolean', False),
    ('cap_input_gate', 'boolean', True),
    ('trainable_initial_state', 'boolean', True),
    ('inter_layer_dropout', 'float', 0.0),
    ('state_dropout', 'float', 0.0),
    # This allows gradual change in the dropout mask. It's kind of in between
    # shared and non-shared masks.
    ('state_dropout_flip_rate', 'float', 0.0),
    ('update_dropout', 'float', 0.0),
    ('cell_clip', 'float', -1.0),

    'objective',
    ('model_average', 'string', 'arithmetic'),
    ('num_training_samples', 'integer', 1),
    ('l2_penalty', 'float', 0.0),
    ('l1_penalty', 'float', 0.0),
    ('activation_norm_penalty', 'float', 0.0),
    ('drop_state_probability', 'float', 0.0),

    'initialization',
    ('embedding_init_factor', 'float', 1.0),
    ('scale_input_embeddings', 'boolean', False),
    ('cell_init_factor', 'float', 1.0),
    ('forget_bias', 'float', 1.0),
    ('output_init_factor', 'float', 1.0),

    'schedule',
    ('steps_per_turn', 'integer', 1000),
    ('print_training_stats_every_num_steps', 'integer', 1000),
    ('turns', 'integer', -1),

    'optimization',
    ('optimizer_type', 'string', 'rmsprop'),
    ('rmsprop_beta2', 'float', 0.999),
    ('rmsprop_epsilon', 'float', 1e-8),
    ('adam_beta1', 'float', 0.9),
    ('adam_beta2', 'float', 0.999),
    ('adam_epsilon', 'float', 1e-8),
    ('batch_size', 'integer', -1),
    ('accum_batch_size', 'integer', -1),
    ('max_grad_norm', 'float', 1.0),
    ('max_time_steps', 'integer', 100),
    ('trigger_averaging_turns', 'integer', -1),
    ('trigger_averaging_at_the_latest', 'integer', -1),

    'learning rate',
    ('learning_rate', 'float', 0.001),
    # TODO(melisgl): Learning rate decay is currently unimplemented.
    #
    # After each optimization step beyond learning_rate_decay_burn_in_steps the
    # effective learning rate is multiplied by learning_rate_decay so that it's
    # equal to learning_rate * pow(decay, max(0, global_step - burn_in_steps)).
    # Also see drop_learning_rate_turns.
    ('learning_rate_decay', 'float', 1.0),
    ('learning_rate_decay_burn_in_steps', 'integer', 0),
    ('drop_learning_rate_turns', 'integer', -1),
    ('drop_learning_rate_multiplier', 'float', 1.0),
    ('drop_learning_rate_at_the_latest', 'integer', -1),

    'early stopping',
    ('early_stopping_turns', 'integer', -1),
    ('early_stopping_rampup_turns', 'integer', 0),
    ('early_stopping_worst_xe_target', 'string', ''),
    ('early_stopping_slowest_rate', 'float', 0.0),

    'cross-validation',
    ('crossvalidate', 'boolean', False),
    ('crossvalidation_folds', 'integer', 10),
    ('crossvalidation_rounds', 'integer', 1),

    'evaluation',
    ('max_training_eval_batches', 'integer', 100),
    ('max_eval_eval_batches', 'integer', -1),
    ('max_test_eval_batches', 'integer', -1),
    ('min_non_episodic_eval_examples_per_stripe', 'integer', 100),
    ('eval_on_test', 'boolean', False),
    ('eval_method', 'string', 'deterministic'),
    ('num_eval_samples', 'integer', 0),
    ('eval_softmax_temperature', 'float', 1.0),
    ('eval_softmax_temperature_estimation_num_tokens', 'integer', 50000),
    ('eval_power_mean_power', 'float', 1.0),
    ('eval_dropout_multiplier', 'float', 1.0),
    ('validation_prediction_file', 'string', ''),
    ('dyneval', 'boolean', False),
    ('dyneval_learning_rate', 'float', 0.001),
    ('dyneval_decay_rate', 'float', 0.02),
    ('dyneval_epsilon', 'float', 1e-5),

    'experiments',
    ('experiment_dir', 'string', '/tmp/lamb'),
    ('save_config', 'boolean', True, 'external'),
    ('config_file', 'string', '', 'external'),
    # Some parameters used to be specified like
    # `--hps=model=lstm,hidden_size=500`, a comma-separated list of assignments.
    ('hps', 'string', '', 'deprecated'),
    # These used to be saved in a sepearate file.
    ('hps_proto_file', 'string', '', 'deprecated'),
    # The old name for config_file.
    ('flags_as_dict', 'string', '', 'deprecated'),

    'checkpoints',
    ('save_checkpoints', 'boolean', True),
    ('load_checkpoint', 'string', '', 'external'),
    ('load_optimizer_state', 'boolean', True, 'external'),
    ('load_averaged', 'boolean', False, 'external'),
    ('use_old_linear_names', 'boolean', False, 'external'),

    'misc',
    ('seed', 'integer', 1),
    ('swap_memory', 'boolean', False),
    ('log_device_placement', 'boolean', False),
    # currently unused
    ('summary_flush_secs', 'integer', 120)
]

FLAGS = flags.FLAGS


def _filter_options(options):
  return [option for option in options
          if not isinstance(option, six.string_types)]


def _define_flags(options):
  for option in _filter_options(options):
    name, type_, default_ = option[:3]
    if type_ == 'boolean':
      flags.DEFINE_boolean(name, default_, '')
    elif type_ == 'integer':
      flags.DEFINE_integer(name, default_, '')
    elif type_ == 'float':
      flags.DEFINE_float(name, default_, '')
    elif type_ == 'string':
      flags.DEFINE_string(name, default_, '')
    elif type_ == 'list_of_ints':
      flags.DEFINE_string(name, default_, '')
    else:
      assert 'Unexpected option type %s' % type_


# Define command-line flags for all options (unless `internal`).
_define_flags(_config_options)


_is_initialized = [False]


def initialize():
  """Override flags from FLAGS.config_file and handle old formats.

  Unless they were explicitly provided on the command line.
  """
  if not _is_initialized[0]:
    assert not (FLAGS.config_file and FLAGS.flags_as_dict), (
        'Both config_file and flags_as_dict were specified.')
    # The deprecated --flags_as_dict used to save some command-line flags as a
    # dict.
    if FLAGS.flags_as_dict:
      logging.info('Handling --flags_as_dict %s', FLAGS.flags_as_dict)
      with tf.gfile.GFile(FLAGS.flags_as_dict, 'r') as f:
        # This contains a single dict.
        args_dict = eval(f.read())  # pylint: disable=eval-used
    if FLAGS.config_file:
      logging.info('Handling --config_file %s', FLAGS.config_file)
      with tf.gfile.GFile(FLAGS.config_file, 'r') as f:
        # This contains a list of bindings.
        args_dict = dict(eval(f.read()))  # pylint: disable=eval-used
    if FLAGS.config_file or FLAGS.flags_as_dict:
      args_dict = _maybe_upgrade_args(args_dict)
      # Update FLAGS with the upgraded values.
      for name, value in args_dict.items():
        if (name not in ['flags_version', 'config_version'] and
            FLAGS[name].using_default_value):
          logging.info('override FLAGS.%s = %r', name, value)
          FLAGS[name].value = value
    _handle_hps()
    _handle_hps_proto_file()
    # Turn off trainable_initial_state for non-episodic mode.
    if not FLAGS.episodic:
      FLAGS.trainable_initial_state = False
    _is_initialized[0] = True


# args_dict comes from either --flags_as_dict or --config_file, either of which
# may be saved using an old format.
def _maybe_upgrade_args(args_dict):
  version = args_dict.get('config_version', 1)
  if version < _config_version:
    logging.info('config file version was %s. Upgrading to %s',
                 version, _config_version)
    if version < 2:
      args_dict['validation_file'] = args_dict.pop('eval_file')
      args_dict['max_time_steps'] = args_dict.pop('max_steps')
      args_dict['steps_per_turn'] = args_dict.pop('steps')
      args_dict['early_stopping_turns'] = args_dict.pop(
          'early_stopping_rounds')
      args_dict['early_stopping_rampup_turns'] = args_dict.pop(
          'early_stopping_rampup_rounds')
      args_dict['print_training_stats_every_num_steps'] = args_dict.pop(
          'print_every')
    if 'averaged_trigger_turns' in args_dict:
      args_dict['trigger_averaging_turns'] = args_dict.pop(
          'averaged_trigger_turns')
    if 'mixture_of_softmaxes_num_components' in args_dict:
      mos_num = args_dict.pop('mixture_of_softmaxes_num_components')
      if mos_num == 1:
        mos_num = 0
      args_dict['mos_num_components'] = mos_num
    if version < 5 and 'hidden_size' in args_dict:
      # FLAGS.hidden_size used to be an int, now it's a string.
      args_dict['hidden_size'] = str(args_dict['hidden_size'])
  else:
    assert version == _config_version, (
        'Unexpected config format version {}'.format(version))
  return args_dict


# No more versions changes, since the corresponding --hps_proto_file is for
# backwards compatibility only.
_hparams_version = 2


_v2_hparam_renames = {
    'intra_layer_dropout': 'inter_layer_dropout',
    'softmax_test_time_temperature': 'eval_softmax_temperature',
    'test_time_power_mean_power': 'eval_power_mean_power',
    'test_time_dropout_multiplier': 'eval_dropout_multiplier',
    'weight_decay': 'l2_penalty',
    'weight_penalty': 'l1_penalty',
    'outer_steps': 'turns',
    'drop_learning_rate_rounds': 'drop_learning_rate_turns',
    'vocab_size': None
}


# Some options used to be specified like `--hps=model=lstm,hidden_size=500`, a
# comma-separated list of assignments. Now, any option can be given via the
# deprecated --hps option.
#
# Error handling is weak, but this is for v1 compatibility only, so that's ok.
def _handle_hps():
  assignments = FLAGS.hps.split(',')
  for assignment in assignments:
    if assignment:
      name, value = assignment.split('=')
      name = _v2_hparam_renames.get(name, name)
      if name and value:
        FLAGS[name].parse(value)
        logging.info('hps: FLAGS.%s = %r', name, FLAGS[name].value)


# There used to be two files in which options were saved. Now there is only one,
# but we must support old saves.
def _handle_hps_proto_file():
  if FLAGS.hps_proto_file:
    hparams_proto = hparam_pb2.HParamDef()
    with tf.gfile.GFile(FLAGS.hps_proto_file) as f:
      text_format.Parse(f.read(), hparams_proto)
    hparams = contrib_training.HParams.from_proto(hparams_proto)
    hparams = _maybe_upgrade_hparams(hparams)
    for name, value in hparams.values().items():
      if FLAGS[name].using_default_value:
        logging.info('hps_proto FLAGS.%s = %r', name, value)
        FLAGS[name].value = value


def _maybe_upgrade_hparams(hparams):
  version = hparams.get('hparams_version', 1)
  if version < _hparams_version:
    logging.info('hps_proto_file version was %s. Upgrading to %s.',
                 version, _hparams_version)
    def rename(old, new):
      # No assignment, delete and readd with new value.
      old_value = hparams.get(old)
      if new and old_value is not None:
        hparams.add_hparam(new, old_value)
      hparams.del_hparam(old)
    if version == 1:
      for old_name, new_name in _v2_hparam_renames.items():
        rename(old_name, new_name)
    if hparams.get('mixture_of_softmaxes_num_components', None):
      rename('mixture_of_softmaxes_num_components', 'mos_num_components')
      if hparams.mos_num_components == 1:
        hparams.mos_num_components = 0
    if hparams.get('hidden_size', None):
      value = str(hparams.get('hidden_size'))
      hparams.del_hparam('hidden_size')
      hparams.add_hparam('hidden_size', value)
  else:
    assert version == _hparams_version, (
        'Unknown hps_proto_file format version {}'.format(version))
  return hparams


# At startup the command-line flags are packaged into a Config object. Some code
# has been refactored to work with Config objects, some code still uses the
# command line arguments directly (as FLAGS.*). In general, we want to minimize
# dependency on FLAGS, and also on Config. Thus relevant parts of Config should
# be extracted and passed as arguments as early as possible.


class Config(object):
  """Flat, mutable configuration with dot notation."""

  def __init__(self, options=()):
    self._options = options
    self._values = {}

  def _find_option(self, name):
    for option in _filter_options(self._options):
      if option[0] == name:
        return option

  def __getattr__(self, name):
    if name in ['_options', '_values', '_find_option']:
      return super(Config, self).__getattribute__(name)
    elif name in self._values:
      return self._values[name]
    else:
      # Lookup the default value.
      option = self._find_option(name)
      if option is None:
        return super(Config, self).__getattribute__(name)
        # raise AttributeError('No config option named {}.'.format(name))
      else:
        return option[2]

  def __setattr__(self, name, value):
    if name in ['_options', '_values', '_find_option']:
      super(Config, self).__setattr__(name, value)
    elif self._find_option(name):
      self._values[name] = value
    else:
      # Add an internal value that doesn't get saved.
      self._options.append((name, 'unknown_type', None, 'internal'))
      self._values[name] = value

  def __getitem__(self, name):
    return getattr(self, name)

  def __setitem__(self, name, value):
    setattr(self, name, value)

  def __contains__(self, name):
    return name in self._values

  def get(self, name, default):
    if name in self:
      return self[name]
    else:
      return default

  def __iter__(self):
    for option in _filter_options(self._options):
      yield option[0]

  def __copy__(self):
    config = self.__class__(copy(self._options))
    config._values = copy(self._values)  # pylint: disable=protected-access
    return config

  def __str__(self):
    s = ''
    for option in self._options:
      if s:
        indent = '  '
      else:
        indent = ' '
      if isinstance(option, six.string_types):
        s += indent + '# ' + option + '\n'
      elif option_visibility(option) != 'internal':
        name = option[0]
        value = self.__getattr__(name)
        s += indent + str((name, value)) + ',\n'
    return '[' + s + ']'

  def save(self, filename):
    with tf.gfile.GFile(filename, 'w') as f:
      f.write(str(self))


def get_config():
  """Return the config in effect.

  Returns:
    A Config containing all the config options (except deprecated or external,
    see _config_options) with values set from command-line arguments.
  """
  options = [option for option in _config_options
             if (isinstance(option, six.string_types) or
                 option_visibility(option) not in ['deprecated', 'external'])]
  config = Config(options)
  # Update the config with the flag values.
  for option in _filter_options(options):
    if option_visibility(option) not in ['deprecated', 'external', 'internal']:
      name = option[0]
      if option[1] == 'list_of_ints':
        if isinstance(FLAGS[name].value, list):
          value = [int(x) for x in FLAGS[name].value]
        else:
          value = [int(x) for x in FLAGS[name].value.split(',')]
      else:
        value = FLAGS[name].value
      config[name] = value
  return config


def handle_config_defaults(config, num_params_fn):
  """Resolve dependencies within `config`.

  In particular, set hidden_size (if -1) according to num_params and make the
  embedding sizes default to the hidden size. Also, handle budgeting: if
  hidden_size is not provided (it is -1), but num_params is, then compute the
  largest possible hidden_size with which the total number of trainable
  parameters does not exceed num_params.

  Args:
    config: The base config. Must have num_params set.
    num_params_fn: A function of one argument a config object. The config passed
      to it is constructed by setting the hidden_size and performing the usual
      defaulting.

  Returns:
    The mutated config.
  """

  # TODO(melisgl): Move this to the tuner code.
  # For ease of specification, tuning ranges are weird. Let's fix them up here.
  if config.sparsity_ratio >= 1.0:
    config.sparsity_ratio = -1.0
  if config.input_embedding_ratio >= 1.0:
    config.input_embedding_ratio = 1.0
  if config.output_embedding_ratio >= 1.0:
    config.output_embedding_ratio = 1.0
  if config.output_embedding_ratio < 0.0:
    config.output_embedding_ratio = config.input_embedding_ratio
  if config.learning_rate_decay > 1.0:
    config.learning_rate_decay = 1.0
  if config.feature_mask_rank < 0:
    config.feature_mask_rank = 0
  if config.inter_layer_dropout < 0.0:
    config.inter_layer_dropout = config.input_dropout
  if config.downprojected_output_dropout < 0.0:
    config.downprojected_output_dropout = config.output_dropout

  # Handle deprecated feature_mask flag.
  if config.feature_mask:
    config.feature_mask_rounds = 2
    config.feature_mask_rank = 0

  # Handle the num_param budget.
  if config.hidden_size in [-1, [-1]]:
    assert config.num_params > -1, (
        'Neither hidden_size nor num_params is specified.')
    config.hidden_size = [_budget_hidden_size(config, num_params_fn)]

  config = _handle_hidden_size_defaults(config)

  # Perform some sanity checks.
  if config.output_embedding_size > config.hidden_size[-1]:
    logging.warn('output_embedding_size %s is greater than '
                 'the hidden size %s', config.output_embedding_size,
                 config.hidden_size[-1])
  if config.share_input_and_output_embeddings:
    assert config.input_embedding_size == config.output_embedding_size

  return config


def _budget_hidden_size(config, num_params_fn):
  """Finds the largest possible hidden size that respects config.num_params.

  Args:
    config: A Config. Must have num_params set.
    num_params_fn: A function of one argument a config object. The config passed
      to it is constructed by setting the hidden_size and performing the usual
      defaulting.

  Returns:
    The largest possible hidden size with which the total number of
    trainable parameters does not exceed config.num_params. Respects
    defaulting rules such as input_embedding_ratio.
  """
  logging.info(
      'Searching for largest possible hidden_size subject to num_params<=%s',
      config.num_params)
  assert config.num_params > 0
  def config_with_hidden_size(hidden_size):
    updated_config = copy(config)
    updated_config.hidden_size = [hidden_size]
    return _handle_hidden_size_defaults(updated_config)
  def is_good(hidden_size):
    n = num_params_fn(config_with_hidden_size(hidden_size))
    good = (n <= config.num_params)
    if n is None:
      logging.info('hidden_size=%s, num_params=OOM BAD', hidden_size)
    elif good:
      logging.info('hidden_size=%s, num_params=%s GOOD', hidden_size, n)
    else:
      logging.info('hidden_size=%s, num_params=%s BAD', hidden_size, n)
    return good, n
  # Double the size until it's too large.
  previous_hidden_size = 1
  hidden_size = 1
  good, n = is_good(hidden_size)
  while good:
    previous_hidden_size = hidden_size
    hidden_size = max(hidden_size+1,
                      int(hidden_size*math.sqrt(1.2*config.num_params / n)))
    good, n = is_good(hidden_size)
  # Bisect the [previous_hidden_size, hidden_size] range.
  def bisect(lower, upper, fn):  # pylint: disable=missing-docstring
    while lower < upper-1:
      # The number of parameters is likely to be at least quadratic in
      # hidden_size. Find the middle point in log space.
      middle = int(math.exp((math.log(upper) + math.log(lower)) / 2))
      middle = min(max(middle, lower+1), upper-1)
      if fn(middle)[0]:
        lower = middle
      else:
        upper = middle
    return lower
  return bisect(previous_hidden_size, hidden_size, is_good)


def _handle_hidden_size_defaults(config):
  """Handle default that depend on hidden_size."""
  last_hidden_size = config.hidden_size[-1]
  for i in six.moves.range(config.num_layers-len(config.hidden_size)):
    config.hidden_size.append(
        max(1, int(last_hidden_size * pow(config.hidden_size_multiplier, i+1))))
  # Now set the actual embedding size if necessary.
  last_hidden_size = config.hidden_size[-1]
  if config.input_embedding_size == -1:
    config.input_embedding_size = max(1, round_to_int(
        config.input_embedding_ratio*last_hidden_size))
  if config.output_embedding_size == -1:
    config.output_embedding_size = max(1, round_to_int(
        config.output_embedding_ratio*last_hidden_size))

  return config


def round_to_int(x):
  return int(round(x))


def flags_as_dict():
  """Return flags that were explicitly provided."""
  dict_ = {}
  for option in _filter_options(_config_options):
    name = option[0]
    if not FLAGS[name].using_default_value:
      dict_[name] = FLAGS[name].value
  return dict_
