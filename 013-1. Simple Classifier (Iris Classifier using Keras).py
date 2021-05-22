"""
Tensorflow가 유명. 근데 다른 라이브러리에 비해 Low level.
이보단 조금 더 고수준 API를 지원하는 Keras를 써볼 것.
"""

# cmd 열고 - conda install keras
"""
# Keras를 사용한 간단한 네트워크.
# 입력층, 은닉층, 출력층이 모두 3차원.
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Sequential은 가장 단순한 모델. 레이어를 순차적으로 쌓을 때 사용.
model = Sequential([
        Dense(3, input_dim=3, activation='sigmoid'), # 은닉층
        Dense(3, activation='sigmoid') # 출력층
        ])

# 모델 학습 준비.
model.compile(
        loss = 'categorical_crossentropy', # 목적함수 정의 
        optimizer='adadelta' # 최적화 알고리즘
        )
"""

# Iris 분류
import sklearn.datasets
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
from sklearn.model_selection import train_test_split

# Iris dataset 준비
ds = sklearn.datasets.load_iris()
X = ds['data']
# Y = pd.get_dummies(ds['target']).as_matrix() # as_matrix() is deprecated.
Y = pd.get_dummies(ds['target']).to_numpy()

# 학습/테스트 셋 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

# 신경망 : 4차원 입력을 받고, 은닉층은 노드 50개로 구성.
# 최종 출력은 3차원. 다중분류 이므로 softmax를 활성함수로 사용.
model = Sequential([
        Dense(50, input_dim=4, activation='sigmoid'),
        Dense(3, activation='softmax')
        ])

# 문제와 활성함수에 맞는 목적함수 설정
model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adadelta'
        )

# 학습 수행
# model.fit(X_train, Y_train, nb_epoch=5) # nb_epochs is deprecated.
model.fit(X_train, Y_train, epochs=5)

# 결과 평가
# proba = model.predict_proba(X_test, batch_size=32) # predict_proba() is deprecated. Please use `model.predict()` instead.
proba = model.predict(X_test, batch_size=32)
pred = pd.Series(proba.flatten())
true = pd.Series(Y_test.flatten())
print('상관계수 : ', pred.corr(true))


