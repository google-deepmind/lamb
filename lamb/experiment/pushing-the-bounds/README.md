This directory is to accompany the [Pushing the bounds of
dropout](https://arxiv.org/abs/1805.09208) paper.

The paper is mostly about how to make predictions with a model trained with
dropout. Use any saved model such as those trained in `../on-the-state/` and
evaluate them with `./test.sh` (in this dir). One difference to `../test.sh` is
that `./test.sh` tunes the optimal evaluation softmax temperature on the
validation set (between 0.8 and 1.0):

    eval_softmax_temperature=-0.8

Also, in addition to deterministic (or 'standard') dropout, it does MC dropout
(the arithmetic averaged variant) with various `eval_dropout_multiplier`s. See
the linked paper for details.

So, assuming there is a saved model in `/tmp/lamb/ptb_10m_lstm_d1/`. Test it
with:

    ./test.sh run some-descriptive-name /tmp/lamb/ptb_10m_lstm_d1/

Thus the model will be evaluated more than once. In the output, the line with
`final test_det_t0.9 xe:` has the test cross-entropy at the optimal softmax
temperature (in this case 0.9). Similarly, `final test_mca_d0.8_t0.9 xe:`
corresponds to the test cross-entropy with `eval_dropout_multiplier=0.8` and
softmax temperature 0.9.
