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

"""Early stopping, checkpointing for training."""

# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


class TrainingMonitor(object):

  def __init__(self, max_turns=None, tuner=None, new_best_fn=None,
               es_turns=None, es_rampup_turns=0, es_ramp_up_from=1,
               es_worst_target=None, es_slowest_rate=None):
    # The max number of turns (evaluations) after which training is stopped.
    self._max_turns = max_turns
    #
    self._tuner = tuner
    # Called when there is an improvement in XE.
    self._new_best_fn = new_best_fn
    # Early stopping: if there is no improvement for this many turns then
    # training is stopped early (i.e. earlier than max_turns).
    self._es_turns = es_turns
    # Gradually ramp up the effective es_turns from es_ramp_up_from during the
    # course of es_rampup_turns so that we pick off unpromising runs quickly.
    self._es_rampup_turns = es_rampup_turns
    self._es_ramp_up_from = es_ramp_up_from
    # If the extrapolated best expected xe is worse than this, then bail out
    # early.
    self._es_worst_target = es_worst_target
    # If the rate of improvement (the decrease in best_xe in the last
    # effective_es_turns()) is less than this, then stop early.
    self._es_slowest_rate = es_slowest_rate
    # These are packaged up as state() for ease of checkpointing.
    self._turn = -1
    self._previous_best_xes = []
    self._best_xe = None
    self._best_xe_turn = None
    self._finished_reason = None
    self._calling_best_fn = False
    self._averaging_triggered = False
    self._metrics = None

  def next_turn(self, evaluator):
    def call_evaluator():
      xe, metrics = evaluator()
      self._metrics = metrics
      return xe

    if self._calling_best_fn or self.finished_reason() is not None:
      # If we loaded a 'best' checkpoint (see the call to self._new_best_fn()
      # below), clear _calling_best_fn so that self._turn gets incremented after
      # the initial evaluation.
      self._calling_best_fn = False
      # Let _finished_reason be recomputed, because early stopping parameters
      # might have changed if we are loaded from a checkpoint.
      self._finished_reason = None
      # Although training may be finished, let's evaluate again for its side
      # effects such as logging results.
      xe = call_evaluator()
      # If evaluation parameters changed, the result may be different now.
      self._maybe_update_best(xe)
      # Parameters (e.g. max_turns) might be different now, so check for
      # termination, but don't report anything to the tuner, because that would
      # lead to duplicated measurements.
      self._update_finished_reason(xe)
    else:
      self._turn += 1
      # Importantly, this is called after turn is sensible.
      xe = call_evaluator()
      # _best_expected_xe assumes that the new result is recorded.
      is_new_best = self._maybe_update_best(xe)
      self._record_best()
      self._update_finished_reason(xe)
      if is_new_best:
        self._calling_best_fn = True
        self._new_best_fn()
        self._calling_best_fn = False

    return self.finished_reason() is None

  # This is -1 before the first next_turn() call, then 0, 1, etc.
  def turn(self):
    return self._turn

  def best_xe(self):
    return self._best_xe

  def best_xe_turn(self):
    return self._best_xe_turn

  def finished_reason(self):
    return self._finished_reason

  def metrics(self):
    # The user provided metrics plus the interesting bits from the state get
    # reported to the tuner.
    metrics = self._metrics.copy()
    metrics['turn'] = self._turn
    metrics['best_xe'] = self._best_xe
    metrics['best_xe_turn'] = self._best_xe_turn
    metrics['finished_reason'] = self._finished_reason
    return metrics

  def state(self):
    return repr(self._state_to_dict())

  def set_state(self, state):
    self._state_from_dict(eval(state))  # pylint: disable=eval-used

  def _state_to_dict(self):
    return {'turn': self._turn,
            'previous_best_xes': self._previous_best_xes,
            'best_xe': self._best_xe,
            'best_xe_turn': self._best_xe_turn,
            'finished_reason': self._finished_reason,
            'calling_best_fn': self._calling_best_fn,
            'averaging_triggered': self._averaging_triggered}

  def _state_from_dict(self, dict_):
    self._turn = dict_['turn']
    self._previous_best_xes = dict_['previous_best_xes']
    self._best_xe = dict_['best_xe']
    self._best_xe_turn = dict_['best_xe_turn']
    self._finished_reason = dict_['finished_reason']
    self._calling_best_fn = dict_['calling_best_fn']
    # Later additions to state. Maintain backward compatibility.
    self._averaging_triggered = dict_.get('averaging_triggered', False)

  def averaging_triggered(self):
    return self._averaging_triggered

  def set_averaging_triggered(self, value):
    self._averaging_triggered = value

  def _maybe_update_best(self, xe):
    if self._best_xe is None or xe < self._best_xe:
      self._best_xe = xe
      self._best_xe_turn = self._turn
      is_new_best = True
    else:
      is_new_best = False
    return is_new_best

  def _record_best(self):
    if self._es_turns:
      self._previous_best_xes.append(self._best_xe)
      max_history = self._es_turns + 2
      if len(self._previous_best_xes) > max_history:
        self._previous_best_xes = self._previous_best_xes[-max_history:]

  def effective_es_turns(self):
    t = self._turn
    d = self._es_turns
    s = self._es_ramp_up_from
    r = self._es_rampup_turns
    if d <= 0:
      return 999999
    elif r == 0:
      return d
    else:
      # Start with s at turn=0 and increase to d at turn=r-d.
      slope = (d - s) / r
      return s + int(slope * min(r, t))

  def _improvement(self):
    es_turns = self.effective_es_turns()
    previous_best_xes = self._previous_best_xes
    if len(previous_best_xes) > es_turns:
      # pylint: disable=invalid-unary-operand-type
      best_xe = self.best_xe()
      improvement = previous_best_xes[-es_turns-1] - best_xe
      assert 0 <= improvement
      return improvement
    else:
      return None

  # Extrapolate from the recent validation results in previous_best_xes,
  def best_expected_xe(self):
    improvement = self._improvement()
    if improvement is not None:
      num_remaining_turns = self._max_turns - self.turn()
      assert num_remaining_turns >= 0
      es_turns = self.effective_es_turns()
      best_xe = self.best_xe()
      # The rate of improvement is decreasing, so this is likely a lower bound.
      return best_xe - improvement*float(num_remaining_turns) / es_turns
    else:
      return -99999.9

  def es_worst_target(self):
    return self._es_worst_target

  def set_es_worst_target(self, value):
    self._es_worst_target = value

  def es_slowest_rate(self):
    return self._es_slowest_rate

  def _rate_of_improvement(self):
    improvement = self._improvement()
    if improvement is not None:
      return improvement / self.effective_es_turns()
    else:
      return None

  def _is_improvement_too_slow(self):
    rate = self._rate_of_improvement()
    return rate is not None and rate < self.es_slowest_rate()

  def _update_finished_reason(self, xe):
    # The tuner only supports numeric values in metrics. Filter out stuff like
    # finished_reason.
    numeric_metrics = self.metrics().copy()
    for key in self.metrics():
      try:
        float(numeric_metrics[key])
      except:  # pylint: disable=bare-except
        numeric_metrics.pop(key)

    if math.isnan(xe):
      if self._tuner:
        self._tuner.report_done(infeasible=True, infeasible_reason='nan')
      self._finished_reason = 'nan'
    elif self._tuner and self._tuner.report_measure(
        xe, global_step=self.turn(), metrics=numeric_metrics):
      # The tuner wants us dead.
      self._finished_reason = 'The tuner said so.'
    elif self._max_turns is not None and self._max_turns <= self.turn():
      self._finished_reason = 'Max turns %d reached.' % (self._max_turns)
    else:
      es_turns = self.effective_es_turns()
      best_expected_xe = self.best_expected_xe()
      es_worst_target = self.es_worst_target()
      # Early stop if there was no improvement in XE for a while.
      if es_turns > 0 and self._best_xe_turn + es_turns < self._turn:
        self._finished_reason = 'No improvement for %s turns.' % (es_turns)
      elif self._is_improvement_too_slow():
        self._finished_reason = 'Improvement too slow (%s<%s).' % (
            self._rate_of_improvement(), self.es_slowest_rate())
      # Extrapolate learning curve and compare to current target.
      elif es_worst_target and es_worst_target < best_expected_xe:
        self._finished_reason = (
            'Best expected XE %f is worse than %f.' %
            (best_expected_xe, es_worst_target))


class LearningRateScheduler(object):

  # TODO(melisgl): Handle decay, decay_burn_in, cyclical learning rates, etc.
  def __init__(self, base_learning_rate, monitor, drop_multiplier=1.0,
               drop_turns=-1, drop_at_turn_at_the_latest=-1):
    assert _read_learning_rate_multiplier() == 1.0, (
        'Found that learning rate is overridden. This is likely unintended.')
    self.base_learning_rate = base_learning_rate
    self.monitor = monitor
    self.drop_multiplier = drop_multiplier
    self.drop_turns = drop_turns
    self.drop_at_turn_at_the_latest = drop_at_turn_at_the_latest
    # These are packaged up as state() for ease of checkpointing.
    self._multiplier = 1.0
    self._last_drop_turn = -1
    self._num_drops = 0

  def state(self):
    return repr(self._state_to_dict())

  def set_state(self, state):
    self._state_from_dict(eval(state))  # pylint: disable=eval-used

  def _state_to_dict(self):
    return {'multiplier': self._multiplier,
            'last_drop_turn': self._last_drop_turn,
            'num_drops': self._num_drops}

  def _state_from_dict(self, dict_):
    self._multiplier = dict_['multiplier']
    self._last_drop_turn = dict_['last_drop_turn']
    self._num_drops = dict_['num_drops']

  def num_drops(self):
    return self._num_drops

  def learning_rate(self):
    turn = self.monitor.turn()
    best_xe_turn = self.monitor.best_xe_turn()
    if ((self.drop_turns > 0 and
         max(best_xe_turn, self._last_drop_turn) + self.drop_turns < turn)
        # Maybe the learning rate hasn't been dropped yet and the latest turn to
        # do so is now:
        or (self._last_drop_turn == -1 and
            self.drop_at_turn_at_the_latest > -1 and
            self.drop_at_turn_at_the_latest <= turn)):
      self._multiplier *= self.drop_multiplier
      self._last_drop_turn = turn
      self._num_drops += 1
    return (self.base_learning_rate *
            self._multiplier *
            _read_learning_rate_multiplier())


def _read_learning_rate_multiplier(filename='/tmp/lamb-override'):
  """Quick hack to allow overriding the learning rate by editing a file."""
  try:
    with open(filename, 'r', encoding='utf-8') as f:
      multiplier = float(f.readline())
  except:  # pylint: disable=bare-except
    multiplier = 1.0
  return multiplier
