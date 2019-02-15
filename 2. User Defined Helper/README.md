# Tensorflow에서 BasicDecoder에 넘겨 줄 수 있는 사용자 정의 Helper Calss를 만들어 보자.
User Defined Helper는 tensorflow.contrib.seq2seq.Helper를 상속받아 구현할 수 있다.

### [목차]
* [왜 User Defined Helper가 필요한가](#왜-User-Defined-Helper가-필요한가)
* [User Defined Helper 만들기](#User-Defined-Helper-만들기)

### [왜 User Defined Helper가 필요한가]
* 기본적으로 TrainingHelper, GreedyEmbeddingHelper, SampleEmbeddingHelper 등을 주로 사용한다.
* 모델에 따라서는 이런 표준적인 Helper로 처리할 수 없는 경우가 있다. 
* 예를 들어, [Tacotron](https://arxiv.org/abs/1703.10135) 모델의 Decoder에서 r(reduction factor, 아래 그림에서는 3)개의 output을 만들어 내고, 그 중 마지막 것을 다음 step의 input으로 넘겨주는 모델에서는 User Defined Helper가 필요하다.
<p align="center"><img width="300" src="tacotron-decoder.png" />  </p>

* train 단계, inference 단계 각각에 맞는 Helper가 필요하다.

### [User Defined Helper 만들기]

