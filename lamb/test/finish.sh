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


# Intended to be sourced after setting all the configuration options.

experiment_dir="$TEST_TMPDIR/${name}"

# Run
source_lib "run.sh" run_par

# Check that the best reported evaluation XE is below a certain
# threshold.
grep_xes() {
  cat "${_experiment_dir}/stderr" |
    sed -rn "s/.*'best_xe': ([0-9]*)\.([0-9]{1,2}).*/\1.\2/p"
}
first_xe=$(grep_xes | head -n 1)
last_xe=$(grep_xes | tail -n 1)
expected_improvement="${expected_improvement:-0.5}"
# check_ge doesn't work with floats, let's do it by hand.
if (( $(echo "$first_xe - $expected_improvement < $last_xe" | bc -l) )); then
  echo "XE went from $first_xe to $last_xe, and that's not a large enough \
improvement ($expected_improvement)."
  exit 1
fi

echo "PASS"
