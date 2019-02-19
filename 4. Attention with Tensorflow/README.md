# Tensorflow에서 Attention Model이 RNN API와 어떻게 연계되어 동작하는지 살펴보자.

## Attention Model
* 대표적인 Attention Model은 `Bahdanau Attention`, `Luong Attention` 등이 있다.
* 이런 Attention Model에 Monotonic한 성질을 더하여 Bahdanau Monotonic Attention, Luong Monotonic Attention이 만들어 질 수도 있다.
* Tensorflow에서는 Attention Model이 `Attention Mechanism` 이라는 개념으로 다루어진다.

![decode](./attentioin-dynamic-rnn-decode.png)

* 위 그림은 Tensorflow Attention API가 이전 tutorial에서 다룬 Tensorflow의 RNN API(BasicDecoder, dynamic_decode 등)와 어떻게 연결되는지 보여주고 있다.
* 이전에 tutorial에서 다룬 BasicRNNCell, BasicLSTMCell, GRUCell 등은 class RNNCell을 상속받아 구현된 class들이다.
* Attention Model을 적용하기 위해서는, 이런 cell들을 대신해서 AttentionWrapper라는 class가 필요한데, 이 AttentionWrapper 또한 RNNCell class를 상속하여 구현된 class이다.
* 
