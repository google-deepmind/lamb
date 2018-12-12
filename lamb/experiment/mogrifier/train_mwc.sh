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

source "$(dirname $0)/../../lib/setup.sh"
source_lib "config/common.sh"
source_lib "config/running.sh"
source_lib "config/mwc.sh"

# Data

lang="${2:-en}"
training_file="${mwc_data_dir}/wiki_${lang}/ptb_format_large/wiki_${lang}.tr.raw.unk"
validation_file="${mwc_data_dir}/wiki_${lang}/ptb_format_large/wiki_${lang}.va.raw.unk"
test_file="${mwc_data_dir}/wiki_${lang}/ptb_format_large/wiki_${lang}.te.raw.unk"

name="$2"
config_file="$3"

source_lib "run.sh" "$1"
