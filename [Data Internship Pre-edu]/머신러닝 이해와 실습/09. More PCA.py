# 최적의 주성분 개수는 어떻게 구하나.
# 1. 그래프 그리기 (PCA Scree plot)
# 2. 설명 가능한 분산량 확인
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

plt.style.use('ggplot')

# 가장 쉬운 Iris를 써서 해보자.
iris = datasets.load_iris()
dataset = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
# 컬럼명 바꾸기
dataset.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# 0,1,2로 된 Class 바꾸기
# 'setosa' 0, 'versicolor' 1, 'virginica' 2
dataset.loc[dataset['Class']==0, 'Class'] = 'Iris-setosa'
dataset.loc[dataset['Class']==1, 'Class'] = 'Iris-versicolor'
dataset.loc[dataset['Class']==2, 'Class'] = 'Iris-virginica'


# X, Y로 나누고
features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
X = dataset.loc[:, features].values
Y = dataset.loc[:, ['Class']].values

# X 정규화
X = StandardScaler().fit_transform(X)

# 고유값을 기준으로 설명할 수 있는 분산량을 확인해보자.
pca = PCA(random_state=1107)
X_p = pca.fit_transform(X)

explainableVar = pd.Series(np.cumsum(pca.explained_variance_ratio_))
print(explainableVar)

# 0개일때 72%, 1개일때 95%, 2개일때 99%를 설명할 수 있음.
# 99%의 설명력을 지닌 2개의 주성분을 선택하는것이 최적임을 알 수 있다.

# 이번엔 그래프를 그려서 확인해보자.
percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
columns = []
for i in range(len(percent_variance)):
    columns.append(f'PC{i+1}')
    
ax = plt.bar(x=range(len(percent_variance)), height=percent_variance, tick_label=columns)
plt.rcParams['figure.figsize'] = (20,15)
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Variance Explained')
plt.title('PCA Scree Plot')
plt.show()