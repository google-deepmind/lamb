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


# This script reproduces the PTB results from "Breaking the Softmax Bottleneck:
# A High-Rank RNN Language Model" (Zhilin Yang, Zihang Dai, Ruslan
# Salakhutdinov, William W. Cohen) without fine-tuning or dynamic evaluation.
#
# Based on https://github.com/zihangdai/mos.

set -e

source "$(dirname $0)/../../lib/setup.sh"
source_lib "config/common.sh"
source_lib "config/running.sh"
source_lib "config/ptb_word.sh"

# Model

share_input_and_output_embeddings=true
input_embedding_size=280
output_embedding_size=280
cap_input_gate=false
input_dropout=0.4
embedding_dropout=0.1
output_dropout=0.4
downprojected_output_dropout=0.29
shared_mask_dropout=true
mos_num_components=15

# Cell

model="lstm"
num_layers=3
lstm_skip_connection=false
hidden_size=960,960,620
inter_layer_dropout=0.225
state_dropout=0.5
tie_forget_and_input_gates=false

# Objective

l2_penalty=8.4e-5 # 1.2e-6*70
drop_state_probability=0.01

# Initialization

forget_bias=0.0

# Schedule

steps_per_turn=100
print_training_stats_every_num_steps=100
turns=8000

# Optimizer

# In the loss, the pytorch code (https://github.com/zihangdai/mos) averages all
# log probabilities in the [batch_size, max_time_steps] matrix, while lamb sums
# the log probabilities over time steps and averages only over the examples in
# the batch. To compensate for that, max_grad_norm, learning_rate and l2_penalty
# had to be adjusted.
max_time_steps=70
max_grad_norm=17.5 # 0.25*70
optimizer_type="sgd"
batch_size=12
learning_rate=0.285 # 20.0/70

# Evaluation hyperparameters

trigger_averaging_turns=50
trigger_averaging_at_the_latest=2000
max_training_eval_batches=20

# Misc

swap_memory=true

source_lib "run.sh" "$@"
