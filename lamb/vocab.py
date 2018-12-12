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

"""Vocabulary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range


class Vocab(object):
  """Immutable reversible mappings from strings to integers."""

  def __init__(self, tokens, unk=u'<UNK>', eos=u'\u25bc'):
    """Create a Vocab object that maps `tokens` to dense indices."""
    self._token_to_index = {}
    self._token_to_frequency = {}
    self._unk = unk
    self._eos = eos
    token_to_index = self._token_to_index
    token_to_frequency = self._token_to_frequency
    # Get the unique tokens from `tokens` that might be a generator.
    for token in tokens:
      token_to_index[token] = True
      token_to_frequency[token] = token_to_frequency.get(token, 0) + 1
    token_to_index[unk] = True
    token_to_index[eos] = True
    # Now that we have a smaller set of tokens, assign ids in sorted
    # order for deterministic encoding.
    self._index_to_token = [None] * len(token_to_index)
    index_to_token = self._index_to_token
    i = 0
    for token in sorted(list(token_to_index)):
      token_to_index[token] = i
      index_to_token[i] = token
      i += 1

  def unk_index(self):
    """Returns the index of the unknown token."""
    return self._token_to_index[self._unk]

  def eos_index(self):
    """Returns the index of the end-of-sentence token."""
    return self._token_to_index[self._eos]

  def token(self, index_):
    """The string whose `index()` is `index_` or an IndexError."""
    return self._index_to_token[index_]

  def __iter__(self):
    """Iterates over tokens in order of indices."""
    for i in range(self.size()):
      yield self.token(i)

  def index_or_unk(self, token):
    """Find the index assigned to `token`.

    Args:
      token: a string.
    Returns:
      The index of `token` or `unk_index()` if it is not in the vocabulary.
    """
    if token in self._token_to_index:
      return self._token_to_index[token]
    else:
      return self.unk_index()

  def size(self):
    """Returns the number of different tokens in the vocabulary."""
    return len(self._index_to_token)

  def decode(self, ids):
    """Decode a sequence of `ids` with `token()`."""
    assert all([0 <= x and x < len(self._index_to_token) for x in ids])
    return [self.token(x) for x in ids]

  def encode(self, tokens, add_eos=True):
    """Encodes a sentence into a list of token indices.

    Args:
      tokens: A list of tokens.
      add_eos: Whether to add the end of sentence token.
    Returns:
      A list of integer token indices where `unk_index()` stands for
      tokens not found in the vocabulary.
    """
    ids = [self.index_or_unk(token) for token in tokens]

    if add_eos:
      ids += [self.eos_index()]

    return ids

  def index_frequency(self, index_):
    return self._token_to_frequency.get(self.token(index_), 0)
