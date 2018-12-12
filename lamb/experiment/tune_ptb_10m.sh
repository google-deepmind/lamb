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
source "$(dirname $0)/../lib/setup.sh"
source_lib "config/common.sh"
source_lib "config/tuning.sh"
source_lib "config/ptb_word_rmsprop.sh"

# Model hyperparameters

num_params=$(million 10)
share_input_and_output_embeddings=true

# Evaluation hyperparameters

eval_softmax_temperature=-0.8

# Tuning parameters

num_workers=60

# Start a number of tuning studies, setting model specific parameters.

model="lstm"
tie_forget_and_input_gates=false
forget_bias=1.0
num_layers=1

tuneables="learning_rate,l2_penalty,
  input_dropout,inter_layer_dropout,state_dropout,
  output_dropout,input_embedding_ratio"
name="$(default_name)_${model}_d${num_layers}"
source_lib "run.sh" "$@"
