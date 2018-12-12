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


set -x

source "$(dirname $0)/start.sh"

# Model hyperparameters

model=lstm
num_layers=2
hidden_size=17,13
output_embedding_size=11
lstm_skip_connection=false

# Optimization hyperparameters

learning_rate=0.2
early_stopping_turns=-1

# Run
source "$(dirname $0)/finish.sh"
previous_xe=$last_xe

# Load checkpoint and check that validation XE is the same.
load_checkpoint="${_experiment_dir}/last"
turns=0
expected_improvement=0.0
source "$(dirname $0)/finish.sh"

if [ "$previous_xe" != "$last_xe" ]; then
  echo "XE was $previous_xe, after reloading checkpoint it became $last_xe."
  exit 1
fi
