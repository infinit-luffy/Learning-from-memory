# R1 — zero-shot retention grid

See `hippoact/TODO_RESULT.md` §R1: the paper's `hard/none` ratio is
misleading here because easy-trained policies score *worse* on clean
backgrounds than on their training distribution, so the denominator is
itself out-of-distribution. `hard/easy` is the meaningful ratio.

```
zero-shot evaluation of trained checkpoints (30 episodes each)
trained on                      eval:none       eval:easy       eval:hard   retention
----------------------------------------------------------------------------------------
dcs-easy-walker-walk        556.6±65.9    853.7±89.0    633.0±85.6        1.137
walker-walk                 916.0±59.6     75.8±7.4      97.1±10.9        0.106
dcs-easy-cheetah-run        118.6±13.6    389.7±84.9    191.8±47.4        1.618
cheetah-run                 443.2±27.9     25.4±4.8      28.6±6.2         0.064

retention = mean(eval on hard) / mean(eval on none), per paper §IV.C.
Rows starting with `dcs-easy-` are the protocol rows (trained on Easy);
clean-trained rows are the contrast (a larger distribution shift).
```
