This directory contains saved configuration files for tuned models from the [On
the state of the art of evaluation in neural language
models](https://arxiv.org/abs/1707.05589) paper. Model weights are not included.

Don't forget to [set up the data](../../README.md).

To train the 1 layer LSTM model of 10m weights on PTB with tuned hyperparameters
(see the paper above):

    ./train_ptb.sh run ptb_10m_lstm_d1/hps_proto

There are separate training script for enwik8 and wikitext-2. The training will
save the model in `/tmp/lamb/ptb_10m_lstm_d1/`. To test the saved model:

    ../test.sh run some-descriptive-name /tmp/lamb/ptb_10m_lstm_d1/
