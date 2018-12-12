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
source_lib "config/wikitext-2_word.sh"

# Model

num_param_millions=35
num_params=$(million ${num_param_millions})
share_input_and_output_embeddings=true
shared_mask_dropout=false

# Cell

model="lstm"
num_layers=2
lstm_skip_connection=true
tie_forget_and_input_gates=false
cap_input_gate=true

# Objective

drop_state_probability=0.01

# Initialization

forget_bias=1.0

# Schedule

steps_per_turn=200
print_training_stats_every_num_steps=200
turns=1000

# Optimizer

optimizer_type="rmsprop"
batch_size=64
max_grad_norm=10.0
max_time_steps=70

# Early stopping

# early_stopping_turns=30
# early_stopping_worst_xe_target=4.4

# Evaluation

max_training_eval_batches=20
eval_softmax_temperature=-0.8

# Tuning parameters

priority=200
num_workers=60

# Misc

swap_memory=true

# Start experiments with dropped learning rate

# drop_learning_rate_turns=100
# drop_learning_rate_multiplier=0.1
# drop_learning_rate_at_the_latest=1600
#
# # feature mask
# tuneables="input_embedding_ratio,learning_rate,l2_penalty,
#   input_dropout,inter_layer_dropout,state_dropout,
#   output_dropout,
#   feature_mask_rounds,feature_mask_rank"
# name="$(default_name)_${num_param_millions}m_${model}_fm_d${num_layers}_rms"
# source_lib "run.sh" "$@"
# 
# # vanilla
# tuneables="input_embedding_ratio,learning_rate,l2_penalty,
#   input_dropout,inter_layer_dropout,state_dropout,
#   output_dropout"
# name="$(default_name)_${num_param_millions}m_${model}_d${num_layers}_rms"
# source_lib "run.sh" "$@"

# Start experiments with averaged optimization

drop_learning_rate_turns=-1
drop_learning_rate_multiplier=1.0
drop_learning_rate_at_the_latest=-1
trigger_averaging_turns=50
trigger_averaging_at_the_latest=800

# feature mask
tuneables="input_embedding_ratio,learning_rate,l2_penalty,
  input_dropout,inter_layer_dropout,state_dropout,
  output_dropout,
  feature_mask_rounds,feature_mask_rank"
name="$(default_name)_${num_param_millions}m_${model}_fm_d${num_layers}_arms"
source_lib "run.sh" "$@"

# vanilla
tuneables="input_embedding_ratio,learning_rate,l2_penalty,
  input_dropout,inter_layer_dropout,state_dropout,
  output_dropout"
name="$(default_name)_${num_param_millions}m_${model}_d${num_layers}_arms"
source_lib "run.sh" "$@"
