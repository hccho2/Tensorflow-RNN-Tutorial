# Tensorflow에서 Attention Model이 RNN API와 어떻게 연계되어 동작하는지 살펴보자.

## Attention Model
* 대표적인 Attention Model은 `Bahdanau Attention`, `Luong Attention` 등이 있다.
* 이런 Attention Model에 Monotonic한 성질을 더하여 Bahdanau Monotonic Attention, Luong Monotonic Attention이 만들어 질 수도 있다.
* Tensorflow에서는 Attention Model이 `Attention Mechanism` 이라는 개념으로 다루어진다.

![decode](./attentioin-dynamic-rnn-decode.png)
