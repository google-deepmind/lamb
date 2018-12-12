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
source_lib "config/wikitext-103.sh"
word_based=true
episodic=false
max_time_steps=35
steps_per_turn=1000
turns=1000
print_training_stats_every_num_steps=1000
early_stopping_turns=30
early_stopping_rampup_turns=60
early_stopping_worst_xe_target=3.5,3.3
drop_learning_rate_turns=26
drop_learning_rate_multiplier=0.1
drop_learning_rate_at_the_latest=900
drop_state_probability=0.01
