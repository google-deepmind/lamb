#!/bin/bash
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


set -e

source "$(dirname $0)/../lib/setup.sh"
source_lib "config/common.sh"
source_lib "config/running.sh"

cmd="$1"
load_checkpoint="$3/best"
config_file="${4:-$3}/config"

save_checkpoints=false
turns=0
min_non_episodic_eval_examples_per_stripe=500000

test_one() {
  local name="$1"
  source_lib "run.sh" "${cmd}"
}

cell="lu"
gpu_type="v100"

eval_on_test=false

eval_method="deterministic"
test_one "$2_det"

# MC dropout evaluation can be a bit better, but it's very slow.
eval_method="arithmetic"
num_eval_samples=200
eval_dropout_multiplier=0.6
test_one "$2_amc$eval_dropout_multiplier"
eval_dropout_multiplier=0.7
test_one "$2_amc$eval_dropout_multiplier"
eval_dropout_multiplier=0.8
test_one "$2_amc$eval_dropout_multiplier"
eval_dropout_multiplier=0.9
test_one "$2_amc$eval_dropout_multiplier"

eval_on_test=true
max_eval_eval_batches=1

eval_method="deterministic"
test_one "$2_test_det"

# MC dropout evaluation can be a bit better, but it's very slow.
eval_method="arithmetic"
num_eval_samples=200
eval_dropout_multiplier=0.6
test_one "$2_test_amc$eval_dropout_multiplier"
eval_dropout_multiplier=0.7
test_one "$2_test_amc$eval_dropout_multiplier"
eval_dropout_multiplier=0.8
test_one "$2_test_amc$eval_dropout_multiplier"
eval_dropout_multiplier=0.9
test_one "$2_test_amc$eval_dropout_multiplier"
