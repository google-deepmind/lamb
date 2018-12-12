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

# Model hyperparameters

model=rhn
num_layers=2
hidden_size=17
output_embedding_size=15
sparsity_ratio=0.5

# Optimization hyperparameters

expected_improvement=0.3
learning_rate=0.2
steps_per_turn=20

# Run
source "$(dirname $0)/finish.sh"
