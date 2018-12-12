<a name="what-is-this"></a>
# What is this?

LAnguage Modelling Benchmarks is to tune and test Tensorflow language models. It
was used in the following papers (alse see [citations](#citations)):

- [On the state of the art of evaluation in neural language models](https://arxiv.org/abs/1707.05589)

  See [./experiment/on-the-state/README.md](./experiment/on-the-state/README.md)
  for more.

- [Pushing the bounds of dropout](https://arxiv.org/abs/1805.09208)

  See
  [./experiment/pushing-the-bounds/README.md](./experiment/pushing-the-bounds/README.md)
  for more.

- [Mogrifier LSTM](https://arxiv.org/abs/1909.01792)

  See [./experiment/mogrifier/README.md](./experiment/mogrifier/README.md) for
  more.

<a name="overview"></a>
# Overview

The default dataset locations are `~/data/<dataset-name>/`. See
`lib/config/{ptb,wikitext-2,wikitext-103,enwik8}.sh` for the defaults.

To train a small LSTM on Penn Treebank, run this script:

    experiment/train_ptb_10m_lstm_d1.sh

In the script, model configuration, data files, etc are specified by setting
variables:

    training_file="ptb.train.txt"
    validation_file="ptb.valid.text"
    model="lstm"
    hidden_size=500

These shell variables are passed as command line arguments to the python
program. These options are documented in the [reference](#reference) section.

To test a trained model:

    experiment/test.sh run "mymodel" "experiment_dir_of_training_run"

In the output, lines with `final valid* xe:':` have the validation set
cross-entropy. Evaluation results are printed as they happen (see the section on
[evaluation](#evaluation)). Lines of special interest in the output are those
with `final {valid,test}` in them. The format is the following:

      final ${dataset}_${eval_method}[_${dropout_multiplier}][_t${softmax_temp}]

For [`eval_method=arithmetic`](#eval_method) with
[`eval_dropout_multiplier=0.8`](#eval_dropout_multiplier) and
[`eval_softmax_temperature=0.9`](#eval_softmax_temperature) results may look
like this after 200 optimization steps and 2 evaluations:

      turn: 2 (eval), step: 200 (opt) (5.29/s)
      final valid_mca_d0.8_t0.9 xe:  5.315
      final test_mca_d0.8_t0.9 xe:  5.289

... except that training runs normally don't have the test set results (see
[`eval_on_test`](#eval_on_test)). Test runs are pretty much training runs with
no optimization steps.

<a name="installation"></a>
# Installation

For example:

    conda create -n tfp3.7 python=3.7 numpy scipy
    conda activate tfp3.7
    conda install cudatoolkit
    conda install cudnn
    conda install tensorflow-gpu=1.15
    conda install tensorflow-probability-gpu=1.15
    conda install tensorflow-probability
    pip install -e <path-to-git-checkout>

<a name="reference"></a>
# Reference

A value given for an option gets converted to the data type corresponding to the
option in question. In the following, options are listed with their data type
and default value (e.g. `model (string, lstm)` means that the variable `model`
has type `string` and default value `lstm`). If there is no default value
listed, then the option is mandatory.

<a name="data"></a>
## Data

- <a name="training_file"></a> `training_file` (string)

  The file with the training data, one line per example. Newlines are translated
  to an end-of-sentence token.

- <a name="validation_file"></a> `validation_file` (string)

  A file of the same format as [`training_file`](#training_file). During
  training, the model is evaluated periodically on data from validation_file. Most
  notably, early stopping and hyperparameter tuning are based on performance on
  this set of examples. This must not be specified when doing cross-validation
  as in that case, the evaluation set is constructed from the training set.

- <a name="test_file"></a> `test_file` (string, '')

  A file of the same format as [`training_file`](#training_file). During
  training, the model evaluated periodically on data from `test_file` and the
  results are logged. As opposed to [`validation_file`](#validation_file), this dataset have
  no affect on training or tuning. The empty string (the default) turns off
  evaluation on the test set.

- <a name="file_encoding"></a> `file_encoding` (string, utf-8)

  The encoding of [`training_file`](#training_file), [`validation_file`](#validation_file)
  and [`test_file`](#test_file).

- <a name="word_based"></a> `word_based` (boolean, false)

  Whether to do word or character based modelling. If word based, lines are
  split at whitespace into tokens. Else, lines are simply split into characters.

- <a name="episodic"></a> `episodic` (boolean, false)

  If true, iterate over examples (lines in the data files) in random order. If
  false, iterate mostly sequentially carrying over model from the previous
  example to the next.

<a name="model"></a>
## Model

- <a name="num_params"></a> `num_params` (float, -1)

  An upper bound on the total number of trainable parameters over all parts of
  the model (including the recurrent cell and input/output embeddings). If this
  is set to a meaningful value (i.e. not -1, the default), then
  [`hidden_size`](#hidden_size) is set to the largest possible value such that
  the parameter budget is not exceeded.

- <a name="share_input_and_output_embeddings"></a> `share_input_and_output_embeddings` (boolean, false)

  Whether the input and output embeddings are the same matrix (transposed) or
  independent (the default). If true, then `input_embedding_size` and
  output_embedding_size must be the same.

- <a name="input_embedding_size"></a> `input_embedding_size` (integer, -1)

  The length of the vector that represents an input token. If -1 (the default),
  then it's determined by [`input_embedding_ratio`](#input_embedding_ratio).

- <a name="output_embedding_size"></a> `output_embedding_size` (integer, -1)

  The length of the vector that represents an output token. If -1 (the default),
  then it's determined by output_embedding_ratio. If - after applying the
  defaulting rules - `output_embedding_size` is not equal to
  [`hidden_size`](#hidden_size), then the cell output is linearly transformed to
  `output_embedding_size` before the final linear transform into the softmax.

- <a name="input_embedding_ratio"></a> `input_embedding_ratio` (float, 1.0)

  If [`input_embedding_size`](#output_embedding_size) is not specified (i.e.
  -1), then it's set to `round(input_embedding_ratio*hidden_size)`.

- <a name="output_embedding_ratio"></a> `output_embedding_ratio` (float, -1.0)

  If [`output_embedding_size`](#output_embedding_size) is not specified (i.e.
  -1), then it's set to `round(output_embedding_ratio*hidden_size)`. The default
  value of -1, makes `output_embedding_ratio` default to the value of
  [`input_embedding_ratio`](#input_embedding_ratio) so that one can tune easily
  with [`share_input_and_output_embeddings`](#share_input_and_output_embeddings)
  `=true`.

- <a name="mos_num_components"></a> `mos_num_components` (integer, 0)

  See [Breaking the softmax bottleneck](https://arxiv.org/abs/1711.03953). The
  default of 0 turns this feature off.

- <a name="embedding_dropout"></a> `embedding_dropout` (float, 0.0)

  The probability that all occurrences of a word are dropped from a batch.

- <a name="token_dropout"></a> `token_dropout` (float, 0.0)

  The probability that a token will be dropped (i.e. the input at that step
  becomes zero). This can be thought of as a version of
  [`embedding_dropout`](#embedding_dropout) that has different masks per time
  step.

- <a name="input_dropout"></a> `input_dropout` (float, 0.0)

  The dropout rate (here and elsewhere, 0 means deterministic operation) for the
  input to the first layer (i.e. just after the input embeddings). This drops
  out individual elements of the embedding vector.

- <a name="output_dropout"></a> `output_dropout` (float, 0.0)

  The dropout rate for just after the cell output.

- <a name="downprojected_output_dropout"></a> `downprojected_output_dropout` (float, -1.0)

  The dropout rate for the projection of the cell output. Only used if
  `output_embedding_size` is different from [`hidden_size`](#hidden_size) or if
  [`mos_num_components`](#mos_num_components)
  is not 1. Defaults to `output_dropout` if set to -1.

- <a name="shared_mask_dropout"></a> `shared_mask_dropout` (boolean, false)

  Whether to use the time same dropout mask for all time steps for
  [`input_dropout`](#input_dropout),
  [`inter_layer_dropout`](#inter_layer_dropout),
  [`output_dropout`](#output_dropout) and
  [`downprojected_output_dropout`](#downprojected_output_dropout).

- <a name="output_once"></a> `output_once` (boolean, true)

  Whether to compute the logits from the cell output in a single operation or
  per time step. The single operation is faster but uses more GPU memory. Also,
  see [`swap_memory`](#swap_memory).

<a name="cell"></a>
### Cell

- <a name="model"></a> `model` (string, lstm)

  One of `lstm`, `rhn` (Recurrent Highway Network), `nas`.

- <a name="num_layers"></a> `num_layers` (integer, 1)

  The number of same-sized LSTM cells stacked on top of each other, or the
  number of processing steps per input an RHN does. Has no effect on NAS.

- <a name="lstm_skip_connection"></a> `lstm_skip_connection` (boolean, true)

  If true, for multi-layer (num_layers>1) LSTMs, the output is computed as the
  sum of the outputs of the individual layers.

- <a name="feature_mask_rounds"></a> `feature_mask_rounds` (integer, 0)

  The Mogrifier LSTM is implemented in terms of the feature masking option. The
  LSTM specific feature masking option involves gating the input and the state
  before they are used for calculating all the other stuff (i.e. `i`, `j`, `o`,
  `f`). This allows input features to be reweighted based on the state, and
  state features to be reweighted based on the input. See the [Mogrifier
  LSTM](https://arxiv.org/abs/1909.01792) paper for details.

  When `feature_mask_rounds` is 0, there is no extra gating in the LSTM.
  When 1<=, the input is gated: `x *= 2*sigmoid(affine(h)))`.
  When 2<=, the state is gated: `h *= 2*sigmoid(affine(x)))`.
  For higher number of rounds, the alternating gating continues.

- <a name="feature_mask_rank"></a> `feature_mask_rank` (integer, 0)

   If 0, the linear transforms described above are full rank, dense matrices. If
   >0, then the matrix representing the linear transform is factorized as the
   product of two low rank matrices (`[*, rank]` and `[rank, *]`). This reduces
   the number of parameters greatly.

- <a name="hidden_size"></a> `hidden_size` (string, "-1")

  A comma-separated list of integers representing the number of units in the
  state of the recurrent cell per layer. Must not be longer than
  [`num_layers`](#num_layers). If it's shorter, then the missing values are
  assumed to be equal to the last specified one. For example, for a 3 layer
  network `"512,256"` results in the first layer having 512 units, the second
  and the third having 256. If "-1" (the default), an attempt is made to deduce
  it from [`num_params`](#num_params) assuming all layers have the same size.

- <a name="layer_norm"></a> `layer_norm` (boolean, false)

  Whether to perform Layer Normalization (currently only implemented for LSTMs).

- <a name="activation_fn"></a> `activation_fn` (string, tf.tanh)

  The non-linearity for the update candidate ('j') and the output ('o') in an
  LSTM, or the output ('h') in an RHN.

- <a name="tie_forget_and_input_gates"></a> `tie_forget_and_input_gates` (boolean, false)

  In an LSTM, whether the input gate ('i') is set to 1 minus the forget gate
  ('f'). In an RHN, whether the transform gate ('t') is set to 1 minus the carry
  gate ('c').

- <a name="cap_input_gate"></a> `cap_input_gate` (boolean, true)

  Whether to cap the input gate at 1-f if
  [`tie_forget_and_input_gates`](#tie_forget_and_input_gates) is off. Currently
  only affects LSTMs. This makes learning more stable, especially at the early
  stages of training.

- <a name="trainable_initial_state"></a> `trainable_initial_state` (boolean, true)

  Whether the initial state of the recurrent cells is allowed to be learnt or is
  set to a fixed zero vector. In non-episodic mode, this switch is forced off.

- <a name="inter_layer_dropout"></a> `inter_layer_dropout` (float, 0.0)

  The input dropout for layers other than the first one. Defaults to no dropout,
  but setting it to -1 makes it inherit [`input_dropout`](#input_dropout). It
  has no effect on RHNs, since the input is not fed to their higher layers.

- <a name="state_dropout"></a> `state_dropout` (float, 0.0)

  This is the dropout rate for the recurrent state from the previous time step
  ('h' in an LSTM, 's' in an RHN). See Yarin Gal's "A Theoretically Grounded
  Application of Dropout in Recurrent Neural Networks". The dropout mask is the
  same for all time steps of a specific example in one batch.

- <a name="update_dropout"></a> `update_dropout` (float, 0.0)

  This is the Recurrent Dropout (see "Recurrent Dropout without Memory Loss")
  rate on the update candidate ('j' in an LSTM, 'h' in an RHN). Should have been
  named Update Dropout.

- <a name="cell_clip"></a> `cell_clip` (float, -1.0)

  If set to a positive value, the cell state ('c' in an LSTM, 's' in an RHN) is
  clipped to the `[-cell_clip, cell_clip]` range after each iteration.

<a name="training"></a>
## Training

<a name="objective"></a>
### Objective

- <a name="model_average"></a> `model_average` (string, arithmetic)

  [Pushing the bounds of dropout](https://arxiv.org/abs/1805.09208) makes the
  point that the actual dropout objective being optimized is a lower bound of
  the true objectives of many different models. If we construct the lower bound
  from multiple samples though (a'la IWAE), the lower bound will get tighter.

  `model_average` is the training time equivalent of `eval_method` and
  determines what kind of model (and consequently, averaging) is to be used. One
  of `geometric`, `power` and `arithmetic`. Only in effect if
  [`num_training_samples`](#num_training_samples) `> 1`.

- <a name="num_training_samples"></a> `num_training_samples` (integer, 1)

  The number of samples from which to compute the objective (see
  [`model_average`](#model_average)). Each training example being presented is
  run through the network `num_training_samples` times so the effective batch
  size is [`batch_size`](#batch_size) `* num_training_samples`. Increasing the
  number of samples doesn't seems to help generalization, though.

- <a name="l2_penalty"></a> `l2_penalty` (float, 0.0)

  The L2 penalty on all trainable parameters.

- <a name="l1_penalty"></a> `l1_penalty` (float, 0.0)

  The L1 penalty on all trainable parameters.

- <a name="activation_norm_penalty"></a> `activation_norm_penalty` (float, 0.0)

  Activation Norm Penalty (Regularizing and optimizing LSTM language models by
  Merity et al).

- <a name="drop_state_probability"></a> `drop_state_probability` (float, 0.0)

  In non-episodic mode, model state is carried over from batch to batch. Not
  feeding back the state with `drop_state_probability` encourages the model to
  work well starting from the zero state which brings it closer to the test
  regime.

<a name="initialization"></a>
### Initialization

- <a name="embedding_init_factor"></a> `embedding_init_factor` (float, 1.0)

  All input embedding weights are initialized with a truncated normal
  distribution with mean 0 and:

      stddev=sqrt(embedding_init_factor/input_embedding_size)

- <a name="scale_input_embeddings"></a> `scale_input_embeddings` (boolean, false)

  This is not strictly an initialization option, but it serves a similar
  purpose. Input embeddings are initialized from a distribution whose variance
  is inversely proportional to [`input_embedding_size`](#input_embedding_size).
  Since every layer in the network is initialized to produce output with
  approximately the same variance as its input, changing the embedding size has
  a potentially strong, undesirable effect on optimization. Set
  `scale_input_embeddings` to `true` to multiply input embeddings by
  `sqrt(input_embedding_size)` to cancel this effect.

  As opposed to just changing `embedding_init_factor`, this multiplication has
  the benefit that the input embedding matrix is of the right scale for use as
  the output embedding matrix should
  [`share_input_and_output_embeddings`](#share_input_and_output_embeddings) be
  turned on.

- <a name="cell_init_factor"></a> `cell_init_factor` (float, 1.0)

  The various weight matrices in the recurrent cell are initialized
  independently (of which there are 8 in an LSTM, 4/2 in an RHN) with

      stddev=sqrt(cell_init_factor/fan_in)

  while biases are initialized with

      stddev=sqrt(cell_init_factor/hidden_size)

- <a name="forget_bias"></a> `forget_bias` (float, 1.0)

  Sometimes initializing the biases of the forget gate ('f') in the LSTM (or
  that of the carry gate ('c') in an RHN) to a small positive value (typically
  1.0, the default) makes the initial phase of optimization faster. Higher
  values make the network forget _less_ of its state over time. With deeper
  architectures and no skip connections (see [`num_layers`](#num_layers) and
  [`lstm_skip_connection`](#lstm_skip_connection)), this may actually make
  optimization harder.

  The value of `forget_bias` is used as the mean of the distribution used for
  initialization with unchanged variance.

- <a name="output_init_factor"></a> `output_init_factor` (float, 1.0)

  If [`share_input_and_output_embeddings`](#share_input_and_output_embeddings)
  is false, then the output projection (also known as the output embeddings) is
  initialized with

      stddev=sqrt(output_init_factor/fan_in)

  If [`share_input_and_output_embeddings`](#share_input_and_output_embeddings)
  is true, then this only affects the linear transform of the cell output (see
  [`output_embedding_size`](#output_embedding_size)).

<a name="schedule"></a>
### Schedule

- <a name="steps_per_turn"></a> `steps_per_turn` (integer, 1000)

  The number of optimization steps between two successive evaluations. After
  this many steps performance is evaluated and logged on the training,
  validation and test sets (if specified). One so called turn consists of
  `steps_per_turn` optimization steps.

- <a name="turns"></a> `turns` (integer)

  The number of evaluations beyond which training cannot continue (also see
  early stopping).

- <a name="print_training_stats_every_num_steps"></a> `print_training_stats_every_num_steps` (integer, 1000)

  Debug printing frequency.

<a name="optimization"></a>
### Optimization

- <a name="optimizer_type"></a> `optimizer_type` (string, rmsprop)

  The optimizer algorithm. One of `rmsprop`, `adam`, `adagrad`, `adadelta` and
  `sgd`.

- <a name="rmsprop_beta2"></a> `rmsprop_beta2` (float, 0.999)

  RMSPROP is actually Adam with `beta1=0.0` so that Adam's highly useful
  correction to the computed statistics is in effect which allows higher initial
  learning rates. Only applies when [`optimizer_type`](#optimizer_type)` is
  `rmsprop`.

- <a name="rmsprop_epsilon"></a> `rmsprop_epsilon` (float, 1e-8)

  Similar to [`adam_epsilon`](#adam_epsilon). Only applies when
  [`optimizer_type`](#optimizer_type) is `rmsprop`.

- <a name="adam_beta1"></a> `adam_beta1` (float, 0.9)

- <a name="adam_beta2"></a> `adam_beta2` (float, 0.999)

- <a name="adam_epsilon"></a> `adam_epsilon`(float, 1e-8)

- <a name="max_grad_norm"></a> `max_grad_norm` (float, 1.0)

  If non-zero, gradients are rescaled so that their norm does not exceed
  `max_grad_norm`.

- <a name="batch_size"></a> `batch_size` (integer)

  Batch size for training. Also, the evaluation batch size unless
  [`min_non_episodic_eval_examples_per_stripe`](#min_non_episodic_eval_examples_per_stripe)
  overrides it.

- <a name="accum_batch_size"></a> `accum_batch_size` (integer, -1)

  The number of examples that are fed to the network at the same time. Set this
  to a divisor of [`batch_size`](#batch_size) to reduce memory usage at the cost
  of possibly slower training. Using `accum_batch_size` does not change the
  results.

- <a name="max_time_steps"></a> `max_time_steps` (integer, 100)

  For episodic operation, examples that have more tokens than this are truncated
  when the training and test files when loaded. For non-episodic operation, this
  is the window size of the truncated backprop.

- <a name="trigger_averaging_turns"></a> `trigger_averaging_turns` (integer, -1)

  The number of turns of no improvement on the validation set, after which
  weight averaging is turned on. Weight averaging is a trivial generalization of
  the idea behind Averaged SGD: it keeps track of the average weights, updating
  the average after each optimization step. Weight averaging does not affect
  training directly, only through evaluation. This feature is an alternative to
  [dropping the learning rate](#drop_learning_rate_turns).

- <a name="trigger_averaging_at_the_latest"></a> `trigger_averaging_at_the_latest` (integer, -1)

  If optimization reaches turn `trigger_averaging_at_the_latest`, then it is
  ensured that averaging is turned on. Set this to be somewhat smaller than
  [`turns`](#turns) so that all runs get at least one drop which should the
  results more comparable.

<a name="learning-rate"></a>
#### Learning rate

- <a name="learning_rate"></a> `learning_rate` (float, 0.001)

- <a name="drop_learning_rate_turns"></a> `drop_learning_rate_turns` (integer, -1)

  If the validation score doesn't improve for `drop_learning_rate_turns` number
  of turns, then the learning rate is multiplied by
  [`drop_learning_rate_multiplier`](#drop_learning_rate_multiplier), possibly
  repeatedly.

- <a name="drop_learning_rate_multiplier"></a> `drop_learning_rate_multiplier` (float, 1.0)

  Set this to a value less than 1.0.

- <a name="drop_learning_rate_at_the_latest"></a> `drop_learning_rate_at_the_latest` (integer, -1)

  If optimization reaches turn
  `drop_learning_rate_multiplier_at_the_latest` without having yet dropped the
  learning rate, then it is dropped regardless of whether the curve is still
  improving or not. Set this to be somewhat smaller than
  [`turns`](#turns) so that all runs get at least one drop which
  should the results more comparable.

<a name="early-stopping"></a>
#### Early stopping

- <a name="early_stopping_turns"></a> `early_stopping_turns` (integer, -1)

  Maximum number of turns without improvement in validation cross-entropy before
  stopping.

- <a name="early_stopping_rampup_turns"></a> `early_stopping_rampup_turns` (integer, 0)

  The effective `early_stopping_turns` starts out at 1 and is increased linearly
  to the specified [`early_stopping_turns`](#early_stopping_turns) in
  `early_stopping_rampup_turns` turns.

- <a name="early_stopping_worst_xe_target"></a> `early_stopping_worst_xe_target` (float, '')

  If the estimated best possible validation cross-entropy (extrapolated from the
  progress made in the most recent
  [`early_stopping_turns`](#early_stopping_turns) (subject to rampup) is worse
  than `early_stopping_worst_xe_target`, then training is stopped. This is
  actually a string of comma separated floats. The first value is in effect when
  the learning rate has not been dropped yet. The second value is effect if it
  has been dropped once and so on. The last element of the list applies to any
  further learning rate drops.

- <a name="early_stopping_slowest_rate"></a> `early_stopping_slowest_rate` (float, 0.0)

  The rate is defined as the average improvement in validation cross-entropy
  over the effective `early_stopping_turns` (see
  [`early_stopping_rampup_turns`](#early_stopping_rampup_turns)). If the rate is
  less than `early_stopping_slowest_rate`, then stop early.

<a name="cross-validation"></a>
### Cross-validation

- <a name="crossvalidate"></a> `crossvalidate` (boolean, false)

  If true, randomly split the training set into
  [`crossvalidation_folds`](#crossvalidation_folds) folds, evaluate performance
  on each and average the cross-entropies. Repeat the entire process for
  [`crossvalidation_rounds`](#crossvalidation_rounds) and average the averages.

- <a name="crossvalidation_folds"></a> `crossvalidation_folds` (integer, 10)

  Number of number of folds to split the training set into.

- <a name="crossvalidation_rounds"></a> `crossvalidation_rounds` (integer, 1)

  If [`crossvalidate`](#crossvalidate), then do this many rounds of
  `crossvalidate_folds`-fold crossvalidation. Set this to a value larger than
  one if the variance of the cross-validation score over the random splits is
  too high.

<a name="evaluation"></a>
## Evaluation

The model being trained is evaluated periodically (see [`turns`](#turns) and
[`steps_per_turn`](#steps_per_turn)) on the validation set (see
[`validation_file`](#validation_file)) and also on the training set (see
[`training_file`](#training_file)). Evaluation on the training set is different
from the loss as it does not include regularization terms such as
[`l2_penalty`](#l2_penalty) and is performed the same way as evaluation on the
validation set (see [`eval_method`](#eval_method)).

To evaluate a saved model one typically wants to do no training, disable saving
of checkpoints and evaluate on the test set which corresponds to this:

    turns=0
    save_checkpoints=false
    eval_on_test=true

Furthermore, [`load_checkpoint`](#load_checkpoint), and in all likelihood
[`config_file`](#config_file) must be set. This is all taken care of by the
`experiment/test.sh` script.

- <a name="max_training_eval_batches"></a> `max_training_eval_batches` (integer, 100)

  When evaluating performance on the training set, it is enough to get a rough
  estimate. If specified, at most `max_training_eval_batches` number of batches
  will be evaluated. Set this to zero, to turn off evaluation on the training
  set entirely. Set it to -1 to evaluate on the entire training set.

- <a name="max_eval_eval_batches"></a> `max_eval_eval_batches` (integer, -1)

  Evaluation can be pretty expensive with large datasets. For expediency, one
  can impose a limit on the number of batches of examples to work with on the
  validation test.

- <a name="max_test_eval_batches"></a> `max_test_eval_batches` (integer, -1)

  Same as [`max_eval_eval_batches`](#max_eval_eval_batches) but for the test
  set.

- <a name="min_non_episodic_eval_examples_per_stripe"></a> `min_non_episodic_eval_examples_per_stripe` (integer, 100)

  By default, evaluation is performed using the training batch size causing each
  "stripe" in a batch to be over rougly dataset_size/batch_size number of
  examples. With a small dataset in a non-episodic setting, that may make the
  evaluation quite pessimistic. This flag ensures that the batch size for
  evaluation is small enought that at least this many examples are processed in
  the same stripe.

- <a name="eval_on_test"></a> `eval_on_test` (boolean, false)

  Even if [`test_file`](#test_file) is provided, evaluation on this test dataset
  is not performed by default. Set this to true to do that. Flipping this switch
  makes it easy to test a model by loading a checkpoint and its saved
  configuration without having to remember what the dataset was.

- <a name="eval_method"></a> `eval_method` (string, deterministic)

  One of `deterministic`, `geometric`, `power` and `arithmetic`. This determines
  how dropout is applied at evaluation.

  - `deterministic` is also known as standard dropout: dropout is turned off at
    evaluation time and a single deterministic pass propagates the expectation
    of each unit through the network.
  - `geometric` performs a renormalized geometric average of predicted
    probabilities over randomly sampled dropout masks.
  - `power` computes the power mean with exponent
    [`eval_power_mean_power`](#eval_power_mean_power).
  - `arithmetic` computes the arithmetic average.

  See [Pushing the bounds of dropout](https://arxiv.org/abs/1805.09208) for a
  more detailed discussion.

- <a name="num_eval_samples"></a> `num_eval_samples` (integer, 0)

  The number of samples to average probabilities over at evaluation time. Needs
  some source of stochasticity (currently only dropout) to be meaningful. When
  it's zero, the model is run in deterministic mode. Training evaluation is
  always performed in deterministic mode for expediency.

- <a name="eval_softmax_temperature"></a> `eval_softmax_temperature` (float, 1.0)

  Set this to a value lower than 1 to smoothen the distribution a bit at
  evaluation time to counter overfitting. Set it to a value between -1 and 0 to
  search for the optimal value between -value and 1 on the validation set. For
  example, `eval_softmax_temperature=-0.8` will search for the optimal
  temperature between 0.8 and 1.0.

- <a name="eval_power_mean_power"></a> `eval_power_mean_power` (float, 1.0)

  The exponent of the renormalized power mean to compute predicted
  probabilities. Only has an effect if [`eval_method=power`](#eval_method).

- <a name="eval_dropout_multiplier"></a> `eval_dropout_multiplier` (float, 1.0)

  At evaluation time all dropout probabilities used for training are multiplied
  by this. Does not affect the [`eval_method=deterministic`](#eval_method) case.
  See [Pushing the bounds of dropout](https://arxiv.org/abs/1805.09208) for
  details.

- <a name="validation_prediction_file"></a> `validation_prediction_file` (string)

  The name of the file where log probabilities of for the validation file are
  written. The file gets superseded by a newer version each time the model is
  evaluated. The file lists tokens and predicted log probabilities on
  alternating lines. Currently only implemented for deterministic
  [evaluation](#eval_method).

- <a name="dyneval"></a> `dyneval` (boolean, false)

  Whether model weights shall be updated at evaluation time (see [Dynamic
  Evaluation of Neural Sequence Models][https://arxiv.org/abs/1709.07432] by
  Krause et al.). This forces batch size at evaluation time to 1 which makes it
  very slow, so turn it is best to leave it off until the final evaluation.

  Whereas RMSProp maintains an online estimate of gradient variance, dynamic
  evaluation bases its estimate on training statistics which are affected by
  [max_training_eval_batches](#max_training_eval_batches) and
  [batch_size](#batch_size).

  Also, when doing dynamic evaluation it might make sense to turn off some
  regularizers such as [l2_penalty](#l2_penalty), or hacks like
  [max_grad_norm](#max_grad_norm).

- <a name="dyneval_learning_rate"></a> `dyneval_learning_rate` (float, 0.001)

  The learning rate for dynamic evaluation.

- <a name="dyneval_decay_rate"></a> `dyneval_decay_rate` (float, 0.02)

  The rate with which weights revert to the _mean_ which is defined as what was
  trained.

- <a name="dyneval_epsilon"></a> `dyneval_epsilon` (float, 1e-5)

  This serves a similar purpose to [`rmsprop_epsilon`](#rmsprop_epsilon), but
  for dynamic evaluation.

<a name="experiments"></a>
## Experiments

- `name` (string, see below) <name="name"></a>

  The name of the experiment. Defaults to the git version concatenated with the
  basename of the script (without the `.sh`). See
  [`experiment_dir`](#experiment_dir).

- <a name="experiment_dir"></a> `experiment_dir` (string, ./ + `$name`)

  Directory for saving configuration, logs and checkpoint files.

  Lamb's git version is saved in `lamb_version` along with any uncommitted
  changes in the checkout (if in a git tree). `stdout` and `stderr` are also
  captured.

  If [`save_checkpoints`](#save_checkpoints) is true, checkpoints are saved
  here. Also see [`save_config`](#save_config).

- <a name="save_config"></a> `save_config` (boolean, true)

  All options are saved in `$experiment_dir/config` except for which it
  doesn't make sense:

  - [`load_checkpoint`](#load_checkpoint)
  - [`load_optimizer_state`](#load_optimizer_state)
  - [`load_averaged`](#load_averaged)
  - [`ensure_new_experiment`](#ensure_new_experiment)
  - [`config_file`](#config_file)
  - [`save_config`](#save_config)

- <a name="ensure_new_experiment"></a> `ensure_new_experiment` (boolean, true)

  If `ensure_new_experiment` is true, a random suffix is appended to
  [`experiment_dir`](#experiment_dir) to ensure the experiment starts afresh.
  When ensure_new_experiment is false and `experiment_dir` exists, the last
  checkpoint will be loaded on startup from that directory.

- <a name="config_file"></a> `config_file` (string, '')

  This is to load `$experiment_dir/config` that gets saved automatically when
  [`save_checkpoints`](#save_checkpoints) is true. It is not needed if one uses
  `experiment/test.sh` for evaluation. If a configuration option is set
  explicitly and is also in the configuration file, then the explicit version
  overrides the one in the configuration file.

<a name="checkpoints"></a>
### Checkpoints

- <a name="save_checkpoints"></a> `save_checkpoints` (boolean, true)

  Whether to save any checkpoints. `save_checkpoints` also affects saving of the
  configuration (see [`config_file`](#config_file).

  If save_checkpoints is true, then two checkpoints are saved:
  `$experiment_dir/best` and `$experiment_dir/last`. If `last` exists, it will
  be loaded automatically on startup and training will continue from that state.
  If that's undesirable, use a different [`experiment_dir`](#experiment_dir) or
  delete the checkpoint manually. The `best` checkpoint corresponds to the best
  validation result seen so far during preriodic model evaluation during
  training.

- <a name="load_checkpoint"></a> `load_checkpoint` (string, '')

  The name of the checkpoint file to load instead of loading
  `$experiment_dir/last` or randomly initializing. Absolute or relative to
  [`experiment_dir`](#experiment_dir).

- <a name="load_optimizer_state"></a> `load_optimizer_state` (boolean, true)

  Set this to `false` to prevent [`load_checkpoint`](#load_checkpoint) from
  attempting to restore optimizer state. This effectively reinitializes the
  optimizer and also allows changing the optimizer type. It does not affect
  automatic loading of the latest checkpoint (see
  [`experiment_dir`](#experiment_dir)).

<a name="misc-options"></a>
### Misc options

- <a name="seed"></a> `seed` (integer, 0)

  The random seed. Both python and tensorflow seeds are initialized with this
  value. Due to non-determinism in tensorflow, training runs are not exactly
  reproducible even with the same seed.

- <a name="swap_memory"></a> `swap_memory` (boolean, false)

  Transparently swap the tensors produced in forward inference but needed for
  back prop from GPU to CPU. This allows training RNNs which would typically not
  fit on a single GPU, but slows things down a bit.

- <a name="log_device_placement"></a> `log_device_placement` (boolean, false)

  Log tensorflow device placement.

<a name="notes"></a>
# Notes

This is not an official Google product.

<a name="citations"></a>
# Citations

- [On the state of the art of evaluation in neural language models](https://arxiv.org/abs/1707.05589)

        @inproceedings{
          melis2018on,
          title={On the State of the Art of Evaluation in Neural Language Models},
          author={G{\'a}bor Melis and Chris Dyer and Phil Blunsom},
          booktitle={International Conference on Learning Representations},
          year={2018},
          url={https://openreview.net/forum?id=ByJHuTgA-},
        }

- [Pushing the bounds of dropout](https://arxiv.org/abs/1805.09208)

        @article{melis2018pushing,
          title={Pushing the bounds of dropout},
          author={Melis, G{\'a}bor and Blundell, Charles and Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}} and Hermann, Karl Moritz and Dyer, Chris and Blunsom, Phil},
          journal={arXiv preprint arXiv:1805.09208},
          year={2018}
        }

- [Mogrifier LSTM](https://arxiv.org/abs/1909.01792)

        @article{melis2020mogrifier,
          title={Mogrifier LSTM},
          author={Melis, G{\'a}bor and Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}} and Blunsom, Phil},
          booktitle={International Conference on Learning Representations},
          year={2020},
          url={https://openreview.net/forum?id=SJe5P6EYvS},
        }
