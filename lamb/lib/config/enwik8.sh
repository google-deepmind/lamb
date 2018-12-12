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

enwik8_data_dir=${enwik8_data_dir:-"${HOME}/data/enwik8/"}
training_file="${enwik8_data_dir}enwik8-training.txt"
validation_file="${enwik8_data_dir}enwik8-valid.txt"
test_file="${enwik8_data_dir}enwik8-test.txt"
file_encoding="CP437"
word_based=false
