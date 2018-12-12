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


# TUNING IS NOT CURRENTLY SUPPORTED IN THE OPEN-SOURCE VERSION. This for
# illustration only.

set -e

# Include definitions of dataset and tuning related variables.
source "$(dirname $0)/../../lib/setup.sh" "$@"
source_lib "config/common.sh"
source_lib "config/tuning.sh"
source_lib "config/ptb_word.sh"

# Model

num_params=$(million 24)
share_input_and_output_embeddings=true
cap_input_gate=false
shared_mask_dropout=true

# Cell

model="lstm"
num_layers=3
lstm_skip_connection=false
tie_forget_and_input_gates=false

# Objective

drop_state_probability=0.01

# Initialization

forget_bias=0.0

# Schedule

steps_per_turn=100
print_training_stats_every_num_steps=100
turns=600

# Optimizer

# In the loss, the pytorch code (https://github.com/zihangdai/mos) averages all
# log probabilities in the [batch_size, max_time_steps] matrix, while lamb sums
# the log probabilities over time steps and averages only over the examples in
# the batch. To compensate for that, max_grad_norm, learning_rate and l2_penalty
# had to be adjusted.
max_time_steps=70
max_grad_norm=10.0
trigger_averaging_turns=25
trigger_averaging_at_the_latest=400

# Early stopping

early_stopping_turns=30
early_stopping_worst_xe_target=4.4

# Evaluation

max_training_eval_batches=20
eval_softmax_temperature=-0.8

# Misc

swap_memory=true

# Tuning parameters

num_workers=60

# SGD
optimizer_type="sgd"
mos_num_components=0
tuneables="batch_size,learning_rate,l2_penalty,
  token_dropout,input_dropout,inter_layer_dropout,state_dropout,
  output_dropout,downprojected_output_dropout,input_embedding_ratio"
name="$(default_name)_${model}_d${num_layers}_asgd"
source_lib "run.sh" "$@"

# RMSPROP
optimizer_type="rmsprop"
mos_num_components=0
tuneables="batch_size,learning_rate,l2_penalty,
  token_dropout,input_dropout,inter_layer_dropout,state_dropout,
  output_dropout,downprojected_output_dropout,input_embedding_ratio"
name="$(default_name)_${model}_d${num_layers}_arms"
source_lib "run.sh" "$@"

# SGD, MoS
optimizer_type="sgd"
mos_num_components=15
tuneables="batch_size,learning_rate,l2_penalty,
  token_dropout,input_dropout,inter_layer_dropout,state_dropout,
  output_dropout,downprojected_output_dropout,input_embedding_ratio"
name="$(default_name)_${model}_d${num_layers}_asgd_mos${mos_num_components}"
source_lib "run.sh" "$@"

# RMSPROP, MoS
optimizer_type="rmsprop"
mos_num_components=15
tuneables="batch_size,learning_rate,l2_penalty,
  token_dropout,input_dropout,inter_layer_dropout,state_dropout,
  output_dropout,downprojected_output_dropout,input_embedding_ratio"
name="$(default_name)_${model}_d${num_layers}_arms_mos${mos_num_components}"
source_lib "run.sh" "$@"
