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

set -e -o pipefail

source googletest.sh

if [ "${base}" = "" ]; then
  source "$(dirname $0)/../lib/setup.sh"
fi
source_lib "config/common.sh"
source_lib "config/running.sh"

training_file="${base}/test/data/corpus.txt"
validation_file="${training_file}"
unset test_file

batch_size=64
max_training_eval_batches=2
max_eval_eval_batches=2
max_test_eval_batches=2
max_time_steps=3
steps_per_turn=5
turns=2

# Misc
use_gpu=false
