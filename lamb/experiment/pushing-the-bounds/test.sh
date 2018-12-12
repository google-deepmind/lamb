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

source "$(dirname $0)/../../lib/setup.sh"
source_lib "config/common.sh"
source_lib "config/running.sh"

saved_args="$1"

save_checkpoints=false
turns=0
min_non_episodic_eval_examples_per_stripe=500000
eval_on_test=true

test_one() {
  local suffix="$1"
  local experiment_dir="$2"
  local name="$(default_name)_${suffix}"
  local config_file="${experiment_dir}/config"
  local load_checkpoint="${experiment_dir}/best"
  source_lib "run.sh" "${saved_args}"
}

name="$2"
experiment_dir="$3"

eval_softmax_temperature=-0.8

eval_method="deterministic"
test_one "det" "${experiment_dir}"

eval_method="arithmetic"
num_eval_samples=200
eval_dropout_multiplier=0.8
test_one "amc$eval_dropout_multiplier" "${experiment_dir}"
