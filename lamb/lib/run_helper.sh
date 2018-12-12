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

escape_cl_arg() {
  printf "%q" "$1"
}

# This command:
#     add_param hps "--" model "X" "escape_cl_arg"
# will add to $hps the line:
#     --model=${model}X
# where ${model} is actually evaluated and transformed by
# escape_cl_arg. See the 'indirect references' shell concept.
add_param() {
  var1="\${$1}"
  prefix=$2
  var2="\${$3}"
  suffix=$4
  val2=$(eval "echo \$$3")
  if [ "$val2" ]; then
    local escape_fn=$5
    if [ "$escape_fn" ]; then
      var2="\$($escape_fn \"$var2\")"
    fi
    eval $1="\"$var1$prefix$3=$var2$suffix\""
  fi
}

add_cl_arg() {
  add_param "$1" "--" "$2" " " "escape_cl_arg"
}

gather_args() {
  ## Populate args (mirroring the structure of README.md). See command line
  ## argument definitions in lamb_flags.py.

  local args=""

  # data
  add_cl_arg args training_file
  add_cl_arg args validation_file
  add_cl_arg args test_file
  add_cl_arg args conditioning_separator
  add_cl_arg args file_encoding
  add_cl_arg args word_based
  add_cl_arg args episodic

  # model
  add_cl_arg args num_params
  add_cl_arg args share_input_and_output_embeddings
  add_cl_arg args input_embedding_size
  add_cl_arg args output_embedding_size
  add_cl_arg args input_embedding_ratio
  add_cl_arg args output_embedding_ratio
  add_cl_arg args embedding_dropout
  add_cl_arg args token_dropout
  add_cl_arg args input_dropout
  add_cl_arg args input_dropout_base
  add_cl_arg args output_dropout
  add_cl_arg args downprojected_output_dropout
  add_cl_arg args shared_mask_dropout
  add_cl_arg args embed_once
  add_cl_arg args output_once

  # cell
  add_cl_arg args model
  add_cl_arg args num_layers
  add_cl_arg args residual_connections
  add_cl_arg args lstm_skip_connection
  add_cl_arg args feature_mask_rounds
  add_cl_arg args feature_mask_rank
  add_cl_arg args sparsity_ratio
  add_cl_arg args overlay_rank
  add_cl_arg args hidden_size
  add_cl_arg args hidden_size_multiplier
  add_cl_arg args layer_norm
  add_cl_arg args activation_fn
  add_cl_arg args tie_forget_and_input_gates
  add_cl_arg args cap_input_gate
  add_cl_arg args mos_num_components
  add_cl_arg args trainable_initial_state
  add_cl_arg args inter_layer_dropout
  add_cl_arg args state_dropout
  add_cl_arg args state_dropout_flip_rate
  add_cl_arg args update_dropout
  add_cl_arg args cell_clip

  # objective
  add_cl_arg args model_average
  add_cl_arg args num_training_samples
  add_cl_arg args l2_penalty
  add_cl_arg args l1_penalty
  add_cl_arg args activation_norm_penalty
  add_cl_arg args drop_state_probability

  # initialization
  add_cl_arg args embedding_init_factor
  add_cl_arg args scale_input_embeddings
  add_cl_arg args cell_init_factor
  add_cl_arg args forget_bias
  add_cl_arg args output_init_factor

  # schedule
  add_cl_arg args steps_per_turn
  add_cl_arg args turns
  add_cl_arg args print_training_stats_every_num_steps
  
  # optimization
  add_cl_arg args optimizer_type
  add_cl_arg args rmsprop_beta2
  add_cl_arg args rmsprop_epsilon
  add_cl_arg args adam_beta1
  add_cl_arg args adam_beta2
  add_cl_arg args adam_epsilon
  add_cl_arg args max_grad_norm
  add_cl_arg args batch_size
  add_cl_arg args accum_batch_size
  add_cl_arg args max_time_steps
  add_cl_arg args trigger_averaging_turns
  add_cl_arg args trigger_averaging_at_the_latest

  # learning rate
  add_cl_arg args learning_rate
  add_cl_arg args learning_rate_decay
  add_cl_arg args learning_rate_decay_burn_in_steps
  add_cl_arg args drop_learning_rate_turns
  add_cl_arg args drop_learning_rate_multiplier
  add_cl_arg args drop_learning_rate_at_the_latest
  
  # early stopping
  add_cl_arg args early_stopping_turns
  add_cl_arg args early_stopping_rampup_turns
  add_cl_arg args early_stopping_worst_xe_target
  add_cl_arg args early_stopping_slowest_rate

  # cross-validation
  add_cl_arg args crossvalidate
  add_cl_arg args crossvalidation_rounds
  add_cl_arg args crossvalidate_max_folds

  # evaluation
  add_cl_arg args max_training_eval_batches
  add_cl_arg args max_eval_eval_batches
  add_cl_arg args max_test_eval_batches
  add_cl_arg args min_non_episodic_eval_examples_per_stripe
  add_cl_arg args eval_on_test
  add_cl_arg args eval_method
  add_cl_arg args num_eval_samples
  add_cl_arg args eval_softmax_temperature
  add_cl_arg args eval_softmax_temperature_estimation_num_tokens
  add_cl_arg args eval_power_mean_power
  add_cl_arg args eval_dropout_multiplier
  add_cl_arg args validation_prediction_file
  add_cl_arg args dyneval
  add_cl_arg args dyneval_learning_rate
  add_cl_arg args dyneval_decay_rate
  add_cl_arg args dyneval_epsilon

  # experiments
  local experiment_dir="${_experiment_dir}"
  add_cl_arg args experiment_dir
  add_cl_arg args save_config
  add_cl_arg args config_file
  add_cl_arg args hps_proto_file # deprecated
  add_cl_arg args flags_as_dict # deprecated

  # checkpoints
  add_cl_arg args save_checkpoints
  add_cl_arg args load_checkpoint
  add_cl_arg args load_optimizer_state
  add_cl_arg args load_averaged
  add_cl_arg args use_old_linear_names

  # Misc flags
  add_cl_arg args seed
  add_cl_arg args swap_memory
  add_cl_arg args logtostderr
  add_cl_arg args log_device_placement
  add_cl_arg args summary_flush_secs

  echo "${args}"
}
