"""
CNN 은 동물의 시각 체계를 모방한 신경망.
계층마다 커널(필터)을 여러개 가지고 있음.
커널을 이용해 2차원 그림 전체를 휩쓸고 다니면서 각 위치에서 다양한 출력값을 계산.
출력값은 커널과 이미지 일부에서 성분별로 곱셈을 수행, 그 결과를 전부 더함. (합성곱convolution 과정)

주로 2차원 입력(이미지)에 많이 사용됨.
"""
"""
Tensor : 다차원 배열or행렬 의미.
 > 입출력으로 갖는 형태를 텐서라고 봐도 됨. (입력은 2차원 배열. 2차원 텐서.)
신경망 내부에선 이 텐서에 다양한 연산을 적용.
 ex) Dense()계층?
  > 입력 텐서를 받고
  > 모든 노드는 입력 텐서의 가중치의 합
  > 입력 텐서가 d차원, 노드가 n개면 가중치 합은 n*d행렬과 d차원 벡터의 행렬곱으로 표현이 가능.
  > 이 결과에 활성함수를 적용.

각 계층은 n*d행렬로 매개변수화되며, 계층 내부에서 일어나는 연산은 대부분 간단한 행렬 연산.
"""

# MNIST dataset : 0~9까지 총 10종류의 숫자를 손으로 쓰고 스캔한 dataset. 각 이미지는 28*28 픽셀, 표본수는 총 7만개.
 ## 2차원 CNN의 경우 실제 입력받는 데이터는 4차원.
 ## 첫번째 차원은 데이터 표본 인덱스, 2번째,3번째 차원은 이미지의 너비와 높이.
 ## 네번째 차원은 이미지의 채널. (흑백은 단일, 컬러는 RGB)
 ## 예를들어, 한번에 4개의 흑백 MNIST 이미지를 학습할 경우 신경망은 (4, 28, 28, 1)크기의 배열(텐서)을 입력받음.
 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D

import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

# MNIST 데이터 다운로드
# data_dict = sklearn.datasets.fetch_mldata('MNIST Original') # 구버전이라 오류남.
data_dict = sklearn.datasets.fetch_openml('mnist_784', version=1, cache=True)
X = data_dict['data']
Y = data_dict['target']

# 학습용/테스트용 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)

# 데이터 4차원 변환
X_train = X_train.values.reshape((63000, 28, 28, 1))
X_test = X_test.values.reshape((7000, 28, 28, 1))
Y_train = pd.get_dummies(Y_train).to_numpy()

nb_samples = X_train.shape[0]
nb_classes = Y_train.shape[1]

# 학습에 사용할 변수
BATCH_SIZE = 16

# 모델 하이퍼 파라미터
KERNEL_WIDTH = 5
KERNEL_HEIGHT = 5
STRIDE = 1
N_FILTERS = 10

# 모델 생성
# Please note, Convolution2D is now Conv2D in the latest version of Keras.
# https://stackoverflow.com/questions/47414651/difference-between-conv2d-and-convolution2d-in-keras
model = Sequential()
"""
model.add(Convolution2D(
        nb_filter = N_FILTERS,
        input_shape = (28, 28, 1),
        nb_row = KERNEL_HEIGHT,
        nb_col = KERNEL_WIDTH,
        subsample = (STRIDE, STRIDE))
        )
"""
model.add(Conv2D(
        filters = N_FILTERS,
        input_shape = (28, 28, 1),
        kernel_size = (KERNEL_HEIGHT, KERNEL_WIDTH),
        strides = (STRIDE, STRIDE)
        ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (5, 5)))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 학습
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print('Start learning...')
model.fit(X_train, Y_train, epochs=10)

# 모델 평가
probs = model.predict_proba(X_test)
preds = model.predict(X_test)
pred_classes = model.predict_classes(X_test)
true_classes = Y_test
(pred_classes == true_classes).sum()

"""
WARNING:tensorflow:7 out of the last 7 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000022338C0C6A8> triggered tf.function retracing. 
Tracing is expensive and the excessive number of tracings could be due to 
(1) creating @tf.function repeatedly in a loop, 
(2) passing tensors with different shapes, 
(3) passing Python objects instead of tensors. 
For (1), please define your @tf.function outside of the loop. 
For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. 
For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

WARNING:tensorflow:From E:/[학교폴더]/[개별공부]/Python Data Science/013-2. MNIST Recognition example.py:96: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
Instructions for updating:
Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   
if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   
if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
"""


