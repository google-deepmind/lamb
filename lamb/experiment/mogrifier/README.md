This directory contains saved configuration files for tuned models from the
[Mogrifier LSTM](https://arxiv.org/abs/1909.01792) paper. Model weights are not
included.

Don't forget to [set up the data](../../README.md).

For example, to train a Mogrifier LSTM with 24M parameters on PTB with tuned
hyperparameters (see the paper above):

    ./train_ptb.sh run train-dir-name config/786252db3825+_tune_ptb_24m_lstm_fm_d2_arms/trial_483/config

There are separate training scripts for other datasets. The `config` directory
holds the best hyperparameters for various model and dataset combinations. The
training will save the model in `./train-dir-name_<unique-id>`. To test the
saved model:

    ../test.sh run test-dir-name ./train-dir-name_<unique-id>/

If training runs out of GPU memory, you may want to decrease `max_time_steps`
(the BPTT window size), but don't expect to reproduce the results that way.
