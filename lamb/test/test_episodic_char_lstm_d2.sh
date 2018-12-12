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


source "$(dirname $0)/start.sh"

training_file="${base}/test/data/add.txt"
validation_file="${training_file}"
expected_improvement="${expected_improvement:-0.2}"

word_based=false
episodic=true
conditioning_separator="="
max_time_steps=40

# Model hyperparameters

model=lstm
num_layers=2
hidden_size=50
num_eval_samples=2

# Optimization hyperparameters

learning_rate=0.01

# Run
source "$(dirname $0)/finish.sh"
