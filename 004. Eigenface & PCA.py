import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
<<<<<<< HEAD
# silhouette_score : 개별 데이터가 가지는 군집화 지표인 실루엣 계수에 대해 평균 계산. 10.7.6 참고.
=======
# silhouette_score : 개별 데이터가 가지는 군집화 지표인 실루엣 계수에 대해 평균 계산. (https://ariz1623.tistory.com/224)
>>>>>>> 25dc8451a7a499b5373c3b9de667142ad237a100
# adjusted_rand_score : 두 클러스터링 간의 유사성 측정값 계산. 확률에 따라 rand score를 재 조정.
# rand_score : 군집화 결과와 정답의 유사성 측정.
from sklearn import metrics

# Data loading
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
# 클래스 40개, 표본 400개, 4096차원, 특징값은 0~1인 실수. 
# faces_data엔 (400, 4096)크기의 data, (400, 64, 64)크기의 images, (400, )크기의 target이 들어감. 
faces_data = datasets.fetch_olivetti_faces()
person_ids, image_array = faces_data['target'], faces_data.images

# Convert (64, 64) images to (4096, ). (64*64=4096)
X = image_array.reshape((len(person_ids), 64*64))


# 군집화 실행
print('## 원본 데이터 군집화 결과')
model = KMeans(n_clusters=40)
model.fit(X)
print('군집화 성능 : ', silhouette_score(X, model.labels_))
print('얼굴 일치율 : ', metrics.adjusted_rand_score(model.labels_, person_ids))

# PCA 실행
print('\n## 주성분 분석 후 군집화 결과')
pca = PCA(25) # 주성분 갯수는 25.
pca.fit(X)
X_reduced = pca.transform(X)
model_reduced = KMeans(n_clusters=40)
model_reduced.fit(X_reduced)
labels_reduced = model_reduced.labels_
print('군집화 성능 : ', silhouette_score(X_reduced, model_reduced.labels_))
print('얼굴 일치율 : ', metrics.adjusted_rand_score(model_reduced.labels_, person_ids))

# 원본 이미지 출력, 확인
sample_face = image_array[0, :, :]
plt.imshow(sample_face)
plt.title('Face Example')
plt.show()


# 아이겐페이스 0 (첫번째 주성분)
eigenface0 = pca.components_[0, :].reshape((64, 64))
plt.imshow(eigenface0)
plt.title("Eigneface 0")
plt.show()

# 아이겐페이스 1 (두번째 주성분)
eigenface1 = pca.components_[1, :].reshape((64, 64))
plt.imshow(eigenface1)
plt.title("Eigenface 1")
plt.show()

## 아이겐페이스 0,1이 차원축소로 인해 원본 이미지보다 형태가 조금 이상해진 것을 볼 수 있음.
