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

# This script runs LAMB.
#
# Usage
# -----
#
# See experiment/*.sh for examples.
#
# Assign values to shell variables of the same name as hyperparameters, command
# line flags and source this script. The single, optional command line argument
# (of the sourcee) is the command which must be "run" in the open source
# version.
#
# setup.py is assumed to have been sourced.
#
# How it works
# ------------
#
# The configuration options (see ../README.md) are gathered from shell variables
# and passed as command line arguments to the binary.

cmd="${1:-run}"

source_lib "run_helper.sh"

_project_dir=${project_dir:-"."}
_experiment_dir="${experiment_dir:-"${_project_dir}/${name}"}"
# If ensure_new_experiment, add a random suffix that makes experiment_dir
# unique.
if [ "${ensure_new_experiment}" != "false" ]; then
  _suffix="$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c5)"
  _experiment_dir="${experiment_dir:-"${_project_dir}/${name}"}_${_suffix}"
  while test -d "${_experiment_dir}"; do
    _suffix="$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c5)"
    _experiment_dir="${experiment_dir:-"${_project_dir}/${name}"}_${_suffix}"
  done
fi

mkdir -p "${_experiment_dir}"

{
  source_lib "describe_version.sh"
}  > >(tee -a "${_experiment_dir}/lamb_version")

{
  if [ "${cmd}" = "run" ]; then
    eval $(echo "python" "${base}/main.py" "$(gather_args)")
  elif [ "${cmd}" = "run_par" ]; then
    eval $(echo "${base}/lamb.par" "$(gather_args)")
  else
    echo "Unsupported command ${cmd}."
    exit 1
  fi
}  > >(tee -a "${_experiment_dir}/stdout") \
  2> >(tee -a "${_experiment_dir}/stderr" >&2)
