# Iris 말고 실습때 썼던 데이터를 써보자.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

plt.style.use('ggplot')

data = pd.read_csv('UseData/wholesale.csv')

pca = PCA(random_state=1107)
data_p = pca.fit_transform(data)

explainableVar = pd.Series(np.cumsum(pca.explained_variance_ratio_))
print(explainableVar)

cumsum = np.cumsum(pca.explained_variance_ratio_)
PCA_num = np.argmax(cumsum >= 0.95) + 1
print('선택할 주성분 수는 : ', PCA_num)

# Scree Plot 그려서 최적의 주성분 갯수 확인해보기.    
"""
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
"""

# 정해진 주성분 갯수(4개)로 새롭게 DBSCAN 해보자.
pca2 = PCA(n_components = PCA_num)
result = pca2.fit_transform(data)
# print(result)

scaler = StandardScaler().fit(result)
result = scaler.transform(result)
# print(result) # 스케일링 된걸 볼 수 있음.

"""
# 확인해볼 그래프? 12 13 14 만 대충 봐보자.
fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

ax1.scatter(result[:, 0], result[:, 1], s=2, color='blue')
ax1.set_xlabel('PCA-1')
ax1.set_ylabel('PCA-2')
ax1.set_title('Wholesale Data - PCA1, 2')

ax2.scatter(result[:, 0], result[:, 2], s=2, color='red')
ax2.set_xlabel('PCA-1')
ax2.set_ylabel('PCA-3')
ax2.set_title('Wholesale Data - PCA1, 3')

ax3.scatter(result[:, 0], result[:, 3], s=2, color='green')
ax3.set_xlabel('PCA-1')
ax3.set_ylabel('PCA-4')
ax3.set_title('Wholesale Data - PCA1, 4')

ax4.scatter(result[:, 1], result[:, 2], s=2, color='royalblue')
ax4.set_xlabel('PCA-2')
ax4.set_ylabel('PCA-3')
ax4.set_title('Wholesale Data - PCA2, 3')

ax5.scatter(result[:, 1], result[:, 3], s=2, color='darkred')
ax5.set_xlabel('PCA-2')
ax5.set_ylabel('PCA-4')
ax5.set_title('Wholesale Data - PCA2, 4')

ax6.scatter(result[:, 2], result[:, 3], s=2, color='darkgreen')
ax6.set_xlabel('PCA-3')
ax6.set_ylabel('PCA-4')
ax6.set_title('Wholesale Data - PCA3, 4')

plt.show()
# 이거 for문 돌려서 깔끔하게 못할까...
"""

# DBSCAN 해보자.
dbsc = DBSCAN(eps=.7, min_samples=20).fit(result)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype=bool)
core_samples[dbsc.core_sample_indices_] = True
# print(labels)
# print(core_samples)

# 색상 테이블 만들어보자
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

for (label, color) in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    cor12 = result[class_member_mask & core_samples]
    ax1.scatter(cor12[:,0], cor12[:,1], color=color, s=2, label='Clustered Data')
    noi12 = result[class_member_mask & ~core_samples]
    ax1.scatter(noi12[:,0], noi12[:,1], color=color, s=2, label='Noise Data') 
    ax1.set_xlabel('PCA-1')
    ax1.set_ylabel('PCA-2')
    ax1.set_title('DBSCAN - PCA1, 2')
    ax1.legend()

    cor13 = result[class_member_mask & core_samples]
    ax2.scatter(cor13[:,0], cor13[:,2], color=color, s=2, label='Clustered Data')
    noi13 = result[class_member_mask & ~core_samples]
    ax2.scatter(noi13[:,0], noi13[:,2], color=color, s=2, label='Noise Data')
    ax2.set_xlabel('PCA-1')
    ax2.set_ylabel('PCA-3')
    ax2.set_title('DBSCAN - PCA1, 3')
    ax2.legend()
    
    cor14 = result[class_member_mask & core_samples]
    ax3.scatter(cor14[:,0], cor14[:,3], color=color, s=2, label='Clustered Data')
    noi14 = result[class_member_mask & ~core_samples]
    ax3.scatter(noi14[:,0], noi14[:,3], color=color, s=2, label='Noise Data')
    ax3.set_xlabel('PCA-1')
    ax3.set_ylabel('PCA-4')
    ax3.set_title('DBSCAN - PCA1, 4')
    ax3.legend()
    
    cor23 = result[class_member_mask & core_samples]
    ax4.scatter(cor23[:,1], cor23[:,2], color=color, s=2, label='Clustered Data')
    noi23 = result[class_member_mask & ~core_samples]
    ax4.scatter(noi23[:,1], noi23[:,2], color=color, s=2, label='Noise Data')
    ax4.set_xlabel('PCA-2')
    ax4.set_ylabel('PCA-3')
    ax4.set_title('DBSCAN - PCA2, 3')
    ax4.legend()
    
    cor24 = result[class_member_mask & core_samples]
    ax5.scatter(cor24[:,1], cor24[:,3], color=color, s=2, label='Clustered Data')
    noi24 = result[class_member_mask & ~core_samples]
    ax5.scatter(noi24[:,1], noi24[:,3], color=color, s=2, label='Noise Data')
    ax5.set_xlabel('PCA-2')
    ax5.set_ylabel('PCA-4')
    ax5.set_title('DBSCAN - PCA2, 4')
    ax5.legend()
    
    cor34 = result[class_member_mask & core_samples]
    ax6.scatter(cor34[:,2], cor34[:,3], color=color, s=2, label='Clustered Data')
    noi34 = result[class_member_mask & ~core_samples]
    ax6.scatter(noi34[:,2], noi34[:,3], color=color, s=2, label='Noise Data')
    ax6.set_xlabel('PCA-3')
    ax6.set_ylabel('PCA-4')
    ax6.set_title('DBSCAN - PCA3, 4')
    ax6.legend()

plt.suptitle('DBSCAN on Wholesale data ($\epsilon = 0.7$, min_samples=20)', fontsize=20)
plt.show()