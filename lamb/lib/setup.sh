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

if [[ "$0" == "$BASH_SOURCE" ]]; then
  echo "This script must be sourced."
  exit 1
fi

base=$(dirname "$BASH_SOURCE")/..

cmd=${1:-"run"}

lib_override_path=

# `source_lib` is like the shell built-in `source`, but allows files in
# `lib_override_path` to shadow those in lamb/lib/.
source_lib() {
  local _name="$1"
  shift
  if [ -d "${lib_override_path}" -a \
       -f "${lib_override_path}/lib/${_name}" ]; then
    source "${lib_override_path}/lib/${_name}" "$@"
  else
    source "${base}/lib/${_name}" "$@"
  fi
}
