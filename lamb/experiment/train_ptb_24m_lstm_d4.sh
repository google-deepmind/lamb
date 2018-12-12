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
source_lib "config/ptb_word_slow.sh"

# Model hyperparameters

model="lstm"
num_params=$(million 24)
share_input_and_output_embeddings=true
tie_forget_and_input_gates=false
cap_input_gate=true
forget_bias=1.0
num_layers=4

# Tuned hyperparameters

learning_rate=0.0033390
l2_penalty=0.000093711
input_dropout=0.68697
inter_layer_dropout=0.31323
state_dropout=0.48479
output_dropout=0.69626

# Evaluation hyperparameters

eval_softmax_temperature=-0.8

source_lib "run.sh" "$@"
