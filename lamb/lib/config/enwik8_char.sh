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

source_lib "config/common.sh"
source_lib "config/enwik8.sh"
# While utf-8 is the actual encoding, for character based modelling
# the literature seems to have settled on bytes as evidenced by
# mentions of a vocabulary size of 205 (it is more than 5000 with
# utf-8).
file_encoding="CP437"
word_based=false
episodic=false
max_time_steps=50
# 400*500=200k optimization steps. With batch size 128 and max_time_steps
# 50, for example, that's about 14 epochs.
steps_per_turn=400
turns=500
print_training_stats_every_num_steps=100
early_stopping_turns=15
early_stopping_rampup_turns=30
early_stopping_worst_xe_target=1.05,0.93,0.92
drop_learning_rate_turns=13
drop_learning_rate_multiplier=0.1
drop_learning_rate_at_the_latest=450
drop_state_probability=0.01
max_eval_eval_batches=500
