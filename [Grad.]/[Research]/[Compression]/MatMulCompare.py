"""
이전부터 들던 의문점 -> (x,y) * (y,z) 가 (x,y) * (y,a) * (a,z) 보다 느린가? (이때 a < (x,y,z))
time을 재봐서 확인. -> Matrix Factorization으로 분해해서 곱연산 vs 원본 곱연산
"""
import numpy as np
import time
import warnings
from sklearn.utils.extmath import randomized_svd # decomposition method 1 : SVD
from sklearn.decomposition import non_negative_factorization # decomposition method 2 : NMF

np.random.seed(62) # fix seed

warnings.filterwarnings('ignore')

# (500,100)(100,80) // (5000,3000)(3000,1000) // (10000,5000)(5000,1000) 
input_matrix = np.random.rand(10000, 5000)
weight_matrix = np.random.rand(5000, 1000)

# print(input_matrix.shape)
# print(weight_matrix.shape)

# 10 // 50 // 100
components = 100

# start = time.process_time()
start_svd_fatcorize = time.process_time()
W_U, W_VT, W_S = randomized_svd(weight_matrix, n_components=components, random_state=62)
W_VT = np.diag(W_VT)
end_svd_factorize = time.process_time()

start_nmf_factorize = time.process_time()
# model = NMF(n_components=components, random_state=62, max_iter=200)
# W_W = model.fit_transform(weight_matrix)
# W_H = model.components_
W_W, W_H, _ = non_negative_factorization(weight_matrix, n_components=components, random_state=62)
end_nmf_factorize = time.process_time()

# print(W_U.shape, W_VT.shape, W_S.shape) # (100, 10) (10, 10) (10, 80)
# print(W_W.shape, W_H.shape) # (100, 10) (10, 80)

# elasped_time = time.process_time() - start
# print(elasped_time)

## time check

start_normal = time.process_time()
result = np.dot(input_matrix, weight_matrix)
end_normal = time.process_time()
# print(result.shape)

# result_svd = np.dot(input_matrix, np.dot(W_U, np.dot(W_VT, W_S)))
start_svd = time.process_time()
result_svd = np.dot(np.dot(np.dot(input_matrix, W_U), W_VT), W_S)
end_svd = time.process_time()
# print(result_svd.shape)

start_nmf = time.process_time()
result_nmf = np.dot(np.dot(input_matrix, W_W), W_H)
end_nmf = time.process_time()
# print(result_nmf.shape)

print('=='*6)
print(f'Normal time : {end_normal - start_normal}')
print(f'   SVD time : {end_svd - start_svd} (Factorizing time : {end_svd_factorize - start_svd_fatcorize})')
print(f'   NMF time : {end_nmf - start_nmf} (Factorizing time : {end_nmf_factorize - start_nmf_factorize})')

# Result
"""
(500, 80)
(500, 80)
(500, 80)
============
Normal time : 0.0015612269999999984
   SVD time : 0.0007163499999999767 (Factorizing time : 0.013159164000000001)
   NMF time : 0.000618501000000049 (Factorizing time : 0.05986941000000001)


(5000, 1000)
(5000, 1000)
(5000, 1000)
============
Normal time : 1.3606363419999994
   SVD time : 0.09083056200000073 (Factorizing time : 0.6108491970000001)
   NMF time : 0.09382024499999986 (Factorizing time : 7.562604783000001)

(10000, 1000)
(10000, 1000)
(10000, 1000)
============
Normal time : 4.5062458240000005
   SVD time : 0.4830305459999984 (Factorizing time : 0.9708892320000001)
   NMF time : 0.4069938810000018 (Factorizing time : 20.639737707000002)
"""