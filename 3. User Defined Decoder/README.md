# 기본적으로 사용하는 BasicDecoder는 

### [목차]
* [User Defined Decoder는 언제 필요한가](#User-Defined-Decoder는-언제-필요한가)


---


### [User Defined Decoder는 언제 필요한가]
* BasicDecoder의 __init__ 함수는 다음과 같은 proto type을 가진다.
<p align="center"><img width="500" src="BasicDecoder.png" />  </p>

* __intit__에 넘겨지는 cell이나 Helper를 표준적인 형식을 벗어나게 customization했다면, BasicDecoder를 사용할 수 없다.
* 좀 더 구체적으로 살펴보자.
* Helper의 next_inputs함수의 proto type은 다음과 같다.
```python
def next_inputs(self, time, outputs, state, sample_ids, name=None):
```
* (time, outputs, state, sample_ids,name)으로 이루어진 argument에 다음과 같이 추가적인 argument가 더해진다고 해보자.
```python
def next_inputs(self, time, outputs, state,  sample_ids, name=None):
```
* 이런 경우에는, BasicDecoder를 customization해야 한다.

* 또 다른 BasciDecoder의 customization 필요성은 cell을  

