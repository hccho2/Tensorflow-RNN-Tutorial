# Tensorflow에서 tf.contrib.seq2seq.dynamic_decode를 어떻게 사용해야 하는지 설명.
아래 그림은 tf.nn.dynamic_rnn과 tf.contrib.seq2seq.dynamic_decode의 입력 구조를 비교한 그림이다.

![decode](./dynamic-rnn-decode.png)
 * Tensorflow에서는 seq2seq(encoder-decoder) 모델을 다룰 수 있는 dynamic_rnn, dynamic_decode를 제공하고 있다.
 * dynamic_rnn은 좀 더 단순한 구조로 되어 있는데, 여기서는 dynamic_decode를 설명한다.
 * cell은 BasicRNNCell,BasicLSTMCell,GRUCell이 올 수 있고, 이런 것들을 쌓은 MultiRNNCell도 올 수 있다.
 * initial_state는 hidden state의 초기값으로 zero_state, encoder의 마지막  hidden state, captioning model에서 image의 feature등이 올 수 있다.
 * TrainingHelper는 training 단계에서 사용하고, GreedyEmbeddingHelper는 inference 단계에서 사용하면 된다.
 * GreedyEmbeddingHelper는 inference에 사용하는 hleper로 전단계의 output의 argmax에 해당하는 결과를 다음 단계의 input으로 전달한다.
 
### [code 설명]
 * 전체 코드는 [RNN-TF-dynamic-decode.py](https://github.com/hccho2/RNN-Tutorial/blob/master/RNN-TF-dynamic-decode.py)에 있고, 이 페이지의 [아래](#full-code)에서도 확인할 수 있다.
 * 이제 코드의 시작부터 차근차근 설명해 보자.
```python
vocab_size = 5
SOS_token = 0
EOS_token = 4

x_data = np.array([[SOS_token, 2, 1, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[2, 1, 2, 3, 2,EOS_token],[3, 1, 2, 3, 1,EOS_token],[ 1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
```
 * 간단한 data로 설명하기 위해, 단어 개수 vocab_size = 5로 설정. 제시된 x_data, y_data를 보면 알 수 있듯이, x_data는 SOS_token으로 시작하고, y_data는 EOS_token으로 끝난다.
 * seq_length는 6이다. batch data들의 길이가 같지 않은 경우가 대부분인데, 이런 경우에는 Null을 도입하여 최대 길이(max_sequence)를 정하고, 뒷부분을 Null로 채워서 길이를 맞춘다. 여기서는 Null을 사용하지 않았다.
 * 실전 data에서는 data file을 읽어, 단어를 숫자로 mapping하고 Null로 padding하는 등의 preprocessing에 많은 시간이 소요될 수 있다.
 * Tensorflow의 data 입력 op인 placeholder를 사용해야하는데, 여기서는 간단함을 위해 사용하지 않는다. 


```python
output_dim = vocab_size
batch_size = len(x_data)
hidden_dim = 6
num_layers = 2
seq_length = x_data.shape[1]
embedding_dim = 8

state_tuple_mode = True
init_state_flag = 0
train_mode = True
```
* output_dim은 RNN cell의 output에 연결되는 FC layer의 출력 dimension이다. 보통의 경우 단어 개수와 동일한 dimension이다. 그래서 output_dim = vocab_size
* batch_size는 추가 설명 불필요^^
* hidden_dim은 말 그대로 RNN cell의 hidden layer size.
* num_layer는 Multi RNN모델에서 RNN layer를 몇 층으로 쌓을지 결정하는 값. 즉, num_layer만큼 LSTM cell을 쌓는다.
* embedding_dim은 각 단어를 몇 차원 vector로 mapping할지 결정하는 변수.
* 나머지 3개 변수(state_tuple_mode,init_state_flag,train_mode)는 코드 상의 옵션을 설정하는 변수로 중요한 것은 아님. 차차 설명.

```python
cells = []
for _ in range(num_layers):
	cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
	cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)    
```
* RNN cell을 num_layer만큼 쌓아야하기 때문에 for loop를 통해서 BasicLSTMCell을 원하는 만큼 쌓았다.
* BasicLSTMCell의 state_is_tuple항목은 c_state와 h_state(m_state라고 하기도 함)를 tuple형태로 관리할 지, 그냥 이어서 하나로 관리할 지 정하는 항목인데, model구조에 영향을 주는 것은 아니다.
* Tensorflow에서는 tuple로 관리할 것을 권장하고 있다.

```python
init = tf.contrib.layers.xavier_initializer()
embedding = tf.get_variable("embedding",shape=[vocab_size,embedding_dim], initializer=init,dtype = tf.float32)
inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim
```

* 각 단어를 embedding vector로 변환할 수 있는 변수를 만든다. vocab_size x embedding_dim만큼의 변수가 필요하다.
* embedding변수가 만들어지면, embedding_lookup을 통해, x_data를 embedding vector로 변환한다.
* 참고로 embedding vector를 만들때 초기값을 아래와 같이 0,1,2,...로 지정하여 embedding vector로 변환이 어떻게 이루어지는지 확인해 볼 수도 있다.
```python
init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1).astype(np.float32) # 아래 embedding의 get_variable에서 shape을 지정하면 안된다.
embedding = tf.get_variable("embedding", initializer=init,dtype = tf.float32)

```

---
* 이제 RNN cell의 hidden state의 초기값을 지정하는 코드를 살펴보자.
```python
if init_state_flag==0:
    initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
else:
    h0 = tf.random_normal([batch_size,hidden_dim]) #실제에서는 적절한 값을 외부에서 받아와야 함.
    if state_tuple_mode: 
        initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)          
    else:
        initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)
```
* hidden state의 초기값은 cell.zero_state(batch_size, tf.float32)와 같이 0으로 지정하는 경우도 있고,
* encoder-decoder 모델에서의 decoder의 hidden state 초기값은 encoder의 마지막 hidden state값을 받아오기도 한다.
* image에 대한 caption을 생성하는 모델에서는 image의 추상화된 feature를 초기값으로 사용할 수도 있다.
* 또한 simple한 Attention 모델에서는 attention vector를 hidden state 초기값으로 전달하기도 한다.
* 우리의 경우, init_state_flag==0인 경우는 0으로 초기화 했고,
* init_state_flag가 0이 아니면, 밖에서 받아온 값으로 초기화해야 하는데, 예를 위해서 h0 = tf.random_normal([batch_size,hidden_dim])를 사용했다.
* LSTM cell에서는 c_state와 h_state가 있기 때문에 각각의 값을 지정해야함.
* 우리는 LSTM cell을 multi로 쌓았기 때문에, 제일 아래 층만 지정된 값을 주고, 나머지 층은 0으로 초기화.
* 제일 아래층에서도 c_state는 0으로 초기화하고, h_state는 h0값으로 초기화 했다.
---
* helper부분을 살펴보자.
```python
if train_mode:
    helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
else:
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)
```
* helper는 input data를 cell에 전달하는 역할을 하는데, training 모드에서는 TrainingHelper를 사용하고, inference 모드에서는 GreedyEmbeddingHelper가 사용된다.
* TrainingHelper의 2번째 argument로 배치의 seq_length를 정해줘야 하는데, 우리는 (필요한 경우 Null을 붙혀) batch속에 있는 각각의 data 길이가 동일(지금의 예에서는 6)하게 만들어 놓았기 때문에, [seq_length]*batch_size로 하면 된다.
* Null을 붙혀 data길이를 맞추었다면, 나중에 loss계산할 때, Null이 붙은 부분의 weight는 0으로 줘서 무시될 수 있도록 하면 된다.
* GreedyEmbeddingHelper는 이전 단계의 output의 argmax에 해당하는 값을 다음 단계의 input으로 전달한다.
* GreedyEmbeddingHelper는 batch개수 만큼의 SOS_token과 EOS_token이 parameter로 넘어간다. EOS_token이 생성될 때까지 RNN 모델이 돌아간다. EOS_token이 생성되지 않으면 무한 루프에 빠질 수 있다.
* 무한 루프에 빠지는 것을 방지하기 위해 아래의 tf.contrib.seq2seq.dynamic_decode에서 maximum_iterations을 지정해 주는 것이 좋다.
---
* 이제 모델의 마지막 부분인 BasicDecoder, dynamic_decode를 살펴보자.
```python
output_layer = Dense(output_dim, name='output_projection')
decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)

```
* output_layer는 RNN cell의 출력값을 받아 연결할 Full Connected Layer를 지정해 준다. output dimension만 정해주면 된다.
* 지금까지의 만든 cell, helper, initial_state, output_layer를 BasicDecoder에 전달하여 decoder를 만들고, 이 decoder를 전달하여 최종적으로 dynamic_decode를 만든다.
---
* 이후의 코드는 Neural Net 모형을 아는 사람은 어렵지 않게 이해할 수 있기 때문에 추가적인 설명은 생략한다.
* 또한, tf.contrib.seq2seq.sequence_loss에서 계산해 주는 loss값과 cross entropy loss를 직접 계산한 값이 일치하는지도 확인하고 있다.
* tf.contrib.seq2seq.sequence_loss의 targets은 one-hot으로 변환되지 않은 값이 전달된다.
* 기타 여러가지 확인할 부분을 출력하는 코드가 추가되어 있다.
---
[추가 설명]
* loss를 계산하는 부분에 대한 설명:
```python
weights = tf.ones(shape=[batch_size,seq_length])
loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)
```
* 우리는 Null을 사용하지 않았기 때문에 모든 batch data의 sequence에 대해서 동일한 가중치 1을 부여했다.
* Null을 사용했다면, Null이 들어가는 부분의 loss가 무시될 수 있도록 weights를 만들어 준다.
```python
weights = tf.to_float(tf.not_equal(y_data, Null))
```








---
### Full CODE

```python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


from tensorflow.python.layers.core import Dense
tf.reset_default_graph()

vocab_size = 5
SOS_token = 0
EOS_token = 4

x_data = np.array([[SOS_token, 2, 1, 2, 3, 2],[SOS_token, 3, 1, 2, 3, 1],[SOS_token, 1, 3, 2, 2, 1]], dtype=np.int32)
y_data = np.array([[2, 1, 2, 3, 2,EOS_token],[3, 1, 2, 3, 1,EOS_token],[ 1, 3, 2, 2, 1,EOS_token]],dtype=np.int32)
print("data shape: ", x_data.shape)


output_dim = vocab_size
batch_size = len(x_data)
hidden_dim = 6
num_layers = 2
seq_length = x_data.shape[1]
embedding_dim = 8

state_tuple_mode = True
init_state_flag = 0
train_mode = True



with tf.variable_scope('test',reuse=tf.AUTO_REUSE) as scope:
    # Make rnn
    cells = []
    for _ in range(num_layers):
        #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=state_tuple_mode)
        cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)    
    #cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)


    #init = np.arange(vocab_size*embedding_dim).reshape(vocab_size,-1).astype(np.float32) # 이경우는 아래의 embedding의 get_variable에서 shape을 지정하면 안된다.
    init = tf.contrib.layers.xavier_initializer()
    embedding = tf.get_variable("embedding",shape=[vocab_size,embedding_dim], initializer=init,dtype = tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, x_data) # batch_size  x seq_length x embedding_dim

    

    if init_state_flag==0:
         initial_state = cell.zero_state(batch_size, tf.float32) #(batch_size x hidden_dim) x layer 개수 
    else:
        h0 = tf.random_normal([batch_size,hidden_dim]) #실제에서는 적절한 값을 외부에서 받아와야 함.
        if state_tuple_mode: 
            initial_state=(tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), h0),) + (tf.contrib.rnn.LSTMStateTuple(tf.zeros_like(h0), tf.zeros_like(h0)),)*(num_layers-1)          
        else:
            initial_state = (tf.concat((tf.zeros_like(h0),h0), axis=1),) + (tf.concat((tf.zeros_like(h0),tf.zeros_like(h0)), axis=1),) * (num_layers-1)
    if train_mode:
        helper = tf.contrib.seq2seq.TrainingHelper(inputs, np.array([seq_length]*batch_size))
    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens=tf.tile([SOS_token], [batch_size]), end_token=EOS_token)

    output_layer = Dense(output_dim, name='output_projection')
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,helper=helper,initial_state=initial_state,output_layer=output_layer)    
    # maximum_iterations를 설정하지 않으면, inference에서 EOS토큰을 만나지 못하면 무한 루프에 빠진다
    outputs, last_state, last_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,output_time_major=False,impute_finished=True,maximum_iterations=10)

    
    Y = tf.convert_to_tensor(y_data)
    weights = tf.ones(shape=[batch_size,seq_length])
    loss =   tf.contrib.seq2seq.sequence_loss(logits=outputs.rnn_output, targets=Y, weights=weights)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)


    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        if train_mode:
            for step in range(100):
                _,l = sess.run([train,loss])
                if step %10 ==0:
                    print("step: {}, loss: {}".format(step,l))
            
            p = sess.run(tf.nn.softmax(outputs.rnn_output)).reshape(-1,output_dim)
            print("loss: {:20.6f}".format(sess.run(loss)))
            print("manual cal. loss: {:0.6f} ".format(np.average(-np.log(p[np.arange(y_data.size),y_data.flatten()]))) )     
        
        print("initial_state: ", sess.run(initial_state))
        print("\n\noutputs: ",outputs)
        o = sess.run(outputs.rnn_output)  #batch_size, seq_length, outputs
        o2 = sess.run(tf.argmax(outputs.rnn_output,axis=-1))
        print("\n",o,o2) #batch_size, seq_length, outputs
    
        print("\n\nlast_state: ",last_state)
        print(sess.run(last_state)) # batch_size, hidden_dim
    
        print("\n\nlast_sequence_lengths: ",last_sequence_lengths)
        print(sess.run(last_sequence_lengths)) #  [seq_length]*batch_size    
        
        print("kernel(weight)",sess.run(output_layer.trainable_weights[0]))  # kernel(weight)
        print("bias",sess.run(output_layer.trainable_weights[1]))  # bias
    
```
