"""
train.csv는 120개의 데이터로 구성
test.csv는 30개의 데이터로 구성
    > species는 모두 비어있는 상황
train.csv를 이용하여 test.csv의 species를 예측하고, 결과를 answer.csv에 저장
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./data/train.csv')

## simple EDA
# print(df.describe(), '\n')
# print(df.info(), '\n')
# print(df['species'].value_counts())     # 40 40 40

# species -> numeric으로 변환
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# X, y로 분리
X = df.drop(columns='species').values
y = df['species'].values

# X 값 scaling -> min-max 간 차이가 극심하지 않은 경우이니 StandardScaler.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# train, test 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# classification 모델 use
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)

# 성능 확인
print(f'test set score: {model_knn.score(X_test, y_test):.3f}', )
print(f'cross validation score: {cross_val_score(model_knn, X_test, y_test, cv=5).mean():.3f}')

# test.csv에 대한 예측 수행
test_df = pd.read_csv('./data/test.csv')

# X값들을 다시 scaling. 이때 scaler는 train에 사용했던 것을 동일하게 사용.
# 동일한 scaler로 scaling 안하는 경우 이상한 예측을 해버리니 반드시 할 것.
X_predict = test_df.drop(columns='species').values
X_predict = scaler.transform(X_predict)

# 예측 수행 및 기존 class 이름으로 재 변환
y_predict = model_knn.predict(X_predict)
test_df['species'] = encoder.inverse_transform(y_predict)

# 10 10 10 으로 split 했지만, model로 예측함에 따라 11 10 9로 나옴.
print(test_df['species'].value_counts())

# 결과 값을 answer.csv로 저장, 이때 species를 제외한 모든 컬럼은 drop.
test_df.drop(columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], inplace=True)
test_df.to_csv('./data/answer.csv', index=False)

"""
# 결정 경계의 확인
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# plot할 2개의 인자 선택
X_plot = X[:, :2]
y_plot = y

# KNN 모델 재학습 (선택한 두 feature만 사용)
    # 기존 model은 4개의 feature를 사용했으나, plot을 위해 2개의 feature만 사용하기 때문.
    # 그리고 선택된 feature에 따라서 결정 경계가 달라지기 때문에...
model_knn_plot = KNeighborsClassifier(n_neighbors=3)
model_knn_plot.fit(X_plot, y_plot)

# 결정 경계 시각화를 위한 meshgrid 생성
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 각 grid point에 대해 예측 수행
Z = model_knn_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 색 설정
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 결정 경계 그리기
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)

# 데이터 plot
scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=cmap_bold, edgecolor='k', s=20)
plt.legend(handles=scatter.legend_elements()[0], labels=list(encoder.classes_), title="Classes") 

# label 추가
plt.xlabel('Feature 1 (sepal_length)')
plt.ylabel('Feature 2 (sepal_width)')
plt.title('KNN Decision Boundary')

plt.show()
"""