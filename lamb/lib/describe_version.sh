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

# We want to know what code was run for an experiment. This prints the git
# version, status and the non-committed diffs, if any.

echo "$(date): Invoking LAMB."
if (which git && git rev-parse --is-inside-work-tree) > /dev/null 2>&1; then
  echo "git version: $(git rev-parse --short HEAD)"
  git --no-pager status
  git --no-pager diff
  git --no-pager diff --cached
fi
