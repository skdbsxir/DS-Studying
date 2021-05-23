"""
대부분의 머신러닝 알고리즘은 데이터가 서로 독립적이라고 가정. 데이터 간 상관관계를 염두에 두지 않음.
하지만 RNN은 연속적인 입력을 받고 과거에 들어온 데이터를 '기억'하는 구조를 가짐.
 > 영상, 음성신호, 문장 등의 데이터에 적합.
 ex) 자연어 처리 기계 번역. (한국어->영어)
  > 한국어 문장의 단어를 순서대로 RNN에 입력.
  > 입력이 끝나면 RNN에는 각종 단어의 정보를 압축한 값이 벡터로 저장.
  > 이 값을 입력으로 하는 또 다른 신경망이 영어 문장을 작성.
  
순환 계층의 종류는 다양. 메모리에 값을 저장하고 꺼내는 방식에 따라 달라지는데, 보통 LSTM을 많이 사용.

단어뿐만 아니라, 시계열 데이터를 이용해 미래 예측도 가능함.
"""
# ex) 바다의 해수면 온도를 측정한 데이터셋을 이용해 과거 11개의 데이터로 다음달 온도를 예측.

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
import statsmodels as sm
import pandas as pd

# Dataset Loading
# numpy 업데이트 하고 나니 데이터를 못불러온다 ㅡㅡ....
# https://www.statsmodels.org/stable/datasets/generated/elnino.html
# https://tedboy.github.io/statsmodels_doc/generated/statsmodels.datasets.elnino.html
# datasets는 statsmodels.api 에 있음.
df = sm.datasets.elnino.load_pandas().data
# df = sm.datasets.get_rdataset('El Nino', 'data')
# df = pd.DataFrame(sm.datasets.elnino.load_pandas().data)
# df['date'] = pd.to_datetime(df.date.apply(lambda x: x.decode('utf-8')))
# df.set_index('date', inplace=True)
X = df.to_numpy()[:, 1:-1]
X = (X - X.min()) / (X.max() - X.min())
# Y = df.to_numpy()[:, -1].values.reshape(61)
Y = df.to_numpy()[:, -1].reshape(61)

# Simple Preprocessing
Y = (Y - Y.min()) / (Y.max() - Y.min())

# train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Modelling
model = Sequential()

# LSTM의 메모리는 20차원 벡터.
# error 발생. 
# https://stackoverflow.com/questions/58479556/notimplementederror-cannot-convert-a-symbolic-tensor-2nd-target0-to-a-numpy
# numpy를 1.19로 downgrade해주면 된다고 함. (현재 1.20.2) 관리자 권한실행 후 pip install numpy==1.19.5
model.add(LSTM(20, input_shape=(11,1)))

# 예측값은 1차원 (스칼라) 값.
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adadelta')

# Learning begin
model.fit(X_train.values.reshape((54, 11, 1)), Y_train, epochs=5)

# 모델 평가
proba = model.predict_proba(X_test.values.reshape((7, 11, 1)), batch_size=32)
pred = pd.Series(proba.flatten())
true = pd.Series(Y_test.flatten())
print('예측값과 실제값의 상관계수 : ', pred.corr(true))