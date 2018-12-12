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

"""Corpus and simple corpus loaders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import random
import sys

from lamb import utils
import numpy as np
import six
# TODO(melisgl): Just for tf.gfile, beh.
import tensorflow.compat.v1 as tf

# pylint: disable=missing-docstring


class Corpus(object):
  """An immutable dataset of instances."""

  def __init__(self, data):
    self._data = data

  def data(self):
    return self._data

  def size(self):
    return len(self._data)

  # Make `dataset + dataset2` and `dataset + sequence` work.
  def __add__(self, other):
    if isinstance(other, Corpus):
      return type(self)(data=self._data + other.data())
    else:
      return type(self)(data=self._data + other)

  def instance_iterator(self, shuffle=False, start=None,
                        max_instances=None, max_epochs=None):
    """Return an iterator over data in the corpus."""
    max_instances = max_instances or sys.maxsize
    max_epochs = max_epochs or sys.maxsize
    data = self._data
    n = len(data)
    permutation = None
    for _ in six.moves.range(max_epochs):
      if shuffle:
        if permutation is None:
          permutation = list(range(n))
        random.shuffle(permutation)
      if start is None:
        offset = 0
      elif start == 'random':
        offset = random.randrange(n)
      elif callable(start):
        offset = start()
      else:
        assert isinstance(start, int)
        offset = start
      num_instances_in_this_epoch = min(n, max_instances)
      for i in six.moves.range(num_instances_in_this_epoch):
        if permutation is None:
          yield data[(offset+i)%n]
        else:
          yield data[permutation[(offset+i)%n]]
      max_instances -= num_instances_in_this_epoch
      if max_instances <= 0:
        break

  # Return the more restrictive of max_instances and max_epochs in
  # units of instances or None if there is no limit.
  def _effective_max_instances(self, max_instances, max_epochs):
    if max_instances is not None and max_epochs is not None:
      return min(max_instances, max_epochs*self.size())
    elif max_instances:
      return max_instances
    elif  max_epochs:
      return max_epochs*self.size()
    else:
      return None

  # `n` is either the corpus size (i.e. the number of examples in the dataset),
  # or the number of examples to iterate over with num_iterators, return the
  # share of the `i`th iterator.
  def _share_of_iterator(self, n, num_iterators, i):
    assert 0 <= i and i < num_iterators
    # TODO(melisgl): Equidistant spacing should take length of
    # individual examples into account.
    if n is not None:
      return (int(((i+1) * n) // num_iterators) -
              int((i * n) // num_iterators))
    else:
      # `n` is infinity (i.e. None). This is typically the case for training
      # iterators (max_instances is None).
      return None

  def ordered_iterators(self, num_iterators,
                        max_instances=None, max_epochs=None):
    """Return a number of iterators as multiple read heads into a corpus.

    In a non-episodic setting, where training examples form a sequence
    (such as the sequence of sentences in a document), one often wants
    to iterate over the dataset in the original order. So far this can
    be done with a simple `instance_iterator`.

    However, when batches of examples are needed, using a single
    iterator would likely lead to consecutive examples being assigned
    to the same batch which is bad for training because the examples
    are then highly correlated, and it also makes techniques that rely
    on carrying over state such as truncated backpropagation
    impossible.

    This function is intended for this batched, non-episodic mode.
    Create one iterator for each stripe in the batch, and let
    `max_instances` and `max_epochs` be automatically distributed
    evenly between all iterators.

    The iterators' starting offset will be drawn randomly and independently from
    each other once at the beginning.

    Args:
      num_iterators: The number of iterators to return.
      max_instances: The total number of examples to iterate over (summed over
        all returned iterators). None means no limit.
      max_epochs: The number of times to iterator over the corpus. If both
        `max_instances` and `max_epochs` are specified, the more restrictive is
        in effect.

    Returns:
      `num_iterators` number of iterators.
    """
    max_instances = self._effective_max_instances(max_instances, max_epochs)
    iterators = []
    for i in six.moves.range(num_iterators):
      iterator_max_instances = self._share_of_iterator(
          max_instances, num_iterators, i)
      if iterator_max_instances is None or iterator_max_instances > 0:
        iterators.append(self.instance_iterator(
            start='random', max_instances=iterator_max_instances))
    return iterators

  def equidistant_iterators(self, num_iterators,
                            random_starts=False, start_jitter=None,
                            max_instances=None, max_epochs=None):
    """Like ordered_iterators but keeps the heads approximately equidistant.

    Args:
      num_iterators: The number of iterators to return.
      random_starts: If True, the starting offset of iterators is shifted by the
        same random number. This is done once when the iterators are created.
      start_jitter: If not None, then after each epoch the starting offsets are
        randomized by adding a random integer from (-start_jitter, start_jitter)
        to them. These jitter offsets are independent from each other.
      max_instances: The total number of examples to iterate over (summed over
        all returned iterators). None means no limit.
      max_epochs: The number of times to iterator over the corpus. If both
        `max_instances` and `max_epochs` are specified, the more restrictive is
        in effect.

    Returns:
      `num_iterators` number of iterators.
    """
    max_instances = self._effective_max_instances(max_instances, max_epochs)
    if max_instances is None:
      n = self.size()
    else:
      n = min(self.size(), max_instances)
    iterators = []
    if random_starts:
      start = random.randrange(n)
    else:
      start = 0
    for i in six.moves.range(num_iterators):
      # Note that if `n` (the corpus size) is equal to max_instances,
      # and max_epochs=1 then all examples will be generated exactly
      # once, because both `start` and `num_instances` are computed by
      # _share_of_iterator.
      iterator_max_instances = self._share_of_iterator(
          max_instances, num_iterators, i)
      if iterator_max_instances is None or iterator_max_instances > 0:
        if start_jitter:
          def start_fn(start=start):
            # pylint: disable=invalid-unary-operand-type
            return (start + random.randint(-start_jitter, start_jitter)) % n
          iterators.append(self.instance_iterator(
              start=start_fn, max_instances=iterator_max_instances))
        else:
          iterators.append(self.instance_iterator(
              start=(start % n), max_instances=iterator_max_instances))
        # Spread the start positions equidistantly.
        start += self._share_of_iterator(n, num_iterators, i)
    return iterators

  def max_sentence_length(self):
    """Returns the maximum number of tokens in the longest sentence."""
    return max([len(datapoint)
                for datapoint in self.instance_iterator(max_epochs=1)]
               # In case the corpus is empty.
               + [0])

  def tokens(self):
    for instance in self.instance_iterator(max_epochs=1):
      for token in instance:
        yield token
      yield u'\u25bc'


def maybe_truncate(seq, n):
  if n is not None and len(seq) > n:
    return seq[:n]
  else:
    return seq


def read_character_based_corpus(filename, encoding='utf-8'):
  with codecs.getreader(encoding)(tf.gfile.GFile(filename, mode='rb')) as f:
    return Corpus([line.rstrip('\n') for line in f])


def read_word_based_corpus(filename, encoding='utf-8'):
  with codecs.getreader(encoding)(tf.gfile.GFile(filename, mode='rb')) as f:
    return Corpus([line.split() for line in f])


def get_episodic_batches(instance_generator, max_batch_size, vocab,
                         num_steps, num_samples=1, max_batches=None,
                         conditioning_separator=None):
  instance_generator = utils.repeat(instance_generator, num_samples)
  max_batch_size *= num_samples
  is_exhausted = False
  while not is_exhausted and (max_batches is None or max_batches > 0):
    if max_batches is not None:
      max_batches -= 1
    instances = []
    for _ in six.moves.range(max_batch_size):
      try:
        instances.append(next(instance_generator))
      except StopIteration:
        is_exhausted = True
        break

    batch_size = len(instances)
    if batch_size == 0:
      break

    if conditioning_separator:
      cond = np.zeros(shape=[num_steps, batch_size], dtype=np.int32)
      cond_len = np.zeros(shape=[batch_size], dtype=np.int64)
    else:
      cond = None
      cond_len = None
    source = np.zeros(shape=[num_steps, batch_size], dtype=np.int32)
    source_len = np.zeros(shape=[batch_size], dtype=np.int64)
    target = np.zeros(shape=[num_steps, batch_size], dtype=np.int32)

    if conditioning_separator:
      # TODO(melisgl): Separate the vocabs.
      conditioning_vocab = vocab
      conditioning_eos = [conditioning_vocab.eos_index()]
      def break_at_separator(seq):
        if conditioning_separator not in seq:
          assert False, 'Conditioning separator {} not found in {}.'.format(
              conditioning_separator, seq)
        pos = seq.index(conditioning_separator)
        return seq[:pos], seq[pos+1:]
    eos = [vocab.eos_index()]
    def emit(batch_index):
      instance_text = instances[batch_index]
      if conditioning_separator:
        conditioning_text, instance_text = break_at_separator(instance_text)
        encoded_conditioning = conditioning_vocab.encode(
            conditioning_text, add_eos=False)
        encoded_conditioning = maybe_truncate(encoded_conditioning, num_steps-1)
        cond_n = len(encoded_conditioning) + 1
        cond[:cond_n, batch_index] = encoded_conditioning + conditioning_eos
        cond_len[batch_index] = cond_n
      encoded_source = vocab.encode(instance_text, add_eos=False)
      encoded_source = maybe_truncate(encoded_source, num_steps-1)
      n = len(encoded_source) + 1
      source[:n, batch_index] = eos + encoded_source
      source_len[batch_index] = n
      target[:n, batch_index] = encoded_source + eos

    for batch_index in range(batch_size):
      emit(batch_index)

    yield (cond, cond_len, source, source_len, target)


def get_non_episodic_batches(instance_generators, vocab,
                             num_steps, num_samples=1, max_batches=None,
                             add_eos=True):
  """Non-episodic."""
  num_generators = len(instance_generators)
  batch_size = num_generators*num_samples
  # Every element produced by an instance generator is a sequence
  # that may be shorter or longer than `num_steps`. We use `sources`
  # as temporary storage for suffixes of those sequences that didn't
  # fit in the previous batch and also as a general staging area.
  #
  # The invariant is that the first element of `sources` is the last
  # element from the previous batch, so these start out effectively
  # empty.
  sources = [[vocab.eos_index()] for _ in six.moves.range(num_generators)]
  # Whether the corresponding iterator is exhausted. Note that
  # `texts` may still be non-empty even if the iterator is
  # exhausted.
  is_exhausteds = [False] * num_generators

  def ensure_window(i):
    # pylint: disable=g-doc-args
    """Move data from the generator to texts[i].

    Ensure that sources[i] has at least num_steps elements available
    or their iterator is exhausted.
    """
    # To produce a source and target sequence of num_steps we need
    # num_steps+1 elements due to the source being shifted.
    while not is_exhausteds[i] and len(sources[i]) <= num_steps:
      try:
        text = next(instance_generators[i])
        encoded_text = vocab.encode(text, add_eos=add_eos)
        sources[i].extend(encoded_text)
      except StopIteration:
        is_exhausteds[i] = True
        break

  def pop_window(i):
    """Extract num_steps (if available)."""
    ensure_window(i)
    # The number of available elements accounting for the special,
    # first one that's the last element from the previous batch.
    n = min(num_steps, len(sources[i]) - 1)
    if n > 0:
      # Extract data.
      encoded_source = sources[i][0:n]
      encoded_target = sources[i][1:n+1]
      # Remove the extracted data, keeping around the last element
      # of the target which will be the first of the next source.
      sources[i] = sources[i][n:]
      return encoded_source, encoded_target
    else:
      return [], []

  def emit_batch():
    source = np.zeros(shape=[num_steps, batch_size], dtype=np.int32)
    source_len = np.zeros(shape=[batch_size], dtype=np.int64)
    target = np.zeros(shape=[num_steps, batch_size], dtype=np.int32)
    emitted_some = False
    for j in six.moves.range(num_generators):
      encoded_source, encoded_target = pop_window(j)
      n = len(encoded_source)
      if n > 0:
        emitted_some = True
      # Repeat it num_samples times.
      for i in six.moves.range(j*num_samples, (j+1)*num_samples):
        source_len[i] = n
        source[:n, i] = encoded_source
        target[:n, i] = encoded_target
    return (emitted_some, source, source_len, target)

  while max_batches is None or max_batches > 0:
    if max_batches is not None:
      max_batches -= 1
    (emitted_some, source, source_len, target) = emit_batch()
    if not emitted_some:
      break
    # No conditioning for non-episodic. Add Nones for the conditioning part.
    yield (None, None, source, source_len, target)


def get_batches(corpus_, vocab, batch_size, num_steps, num_samples=1,
                episodic=None, deterministic=None, equidistant=True,
                max_instances=None, max_epochs=None, max_batches=None,
                conditioning_separator=None):
  if episodic:
    return get_episodic_batches(
        corpus_.instance_iterator(
            shuffle=not deterministic,
            max_instances=max_instances, max_epochs=max_epochs),
        batch_size, vocab, num_steps, num_samples=num_samples,
        max_batches=max_batches, conditioning_separator=conditioning_separator)
  else:
    if deterministic:
      iterators = corpus_.equidistant_iterators(
          batch_size, random_starts=False, start_jitter=None,
          max_instances=max_instances, max_epochs=max_epochs)
    elif equidistant:
      iterators = corpus_.equidistant_iterators(
          batch_size, random_starts=True, start_jitter=100,
          max_instances=max_instances, max_epochs=max_epochs)
    else:
      iterators = corpus_.ordered_iterators(
          batch_size, max_instances=max_instances, max_epochs=max_epochs)
    return get_non_episodic_batches(
        iterators, vocab, num_steps, num_samples=num_samples,
        max_batches=max_batches)
