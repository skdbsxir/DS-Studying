from __future__ import division
from __future__ import print_function

import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

tf.logging.set_verbosity(tf.logging.ERROR)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# Default data is 'cora'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# print(adj) # 1이 기록된 위치 (row, col)를 가지고 있다.
# print(type(adj)) # csr_matrix
# print(adj.shape) # (2708, 2708)
# print(adj.shape[0])
# print(adj.getnnz()) # 10556

# print(features) # adj처럼 기록된 위치 (row, col)을 가지고 있지만, type이 lil_matrix

# print(y_train.shape) # sparse binary ndarray. (2708, 7)
# print(train_mask.shape) # boolean ndarray. (2708,)

# features = preprocess_features(features)
# print(features) # tuple => (ndarray, ndarray, tuple)
# print(type(features[0])) 
# print(type(features[1]))
# print(type(features[2]))

"""
# Sample Testing
# For check some operations
"""

# Sample data from slide (5*5 undirected graph)
# A = np.array([
#     [0,1,1,0,1],
#     [1,0,0,1,1],
#     [1,0,0,0,1],
#     [0,1,0,0,0],
#     [1,1,1,0,0]
# ])

# D = np.array([
#     [3,0,0,0,0],
#     [0,3,0,0,0],
#     [0,0,2,0,0],
#     [0,0,0,1,0],
#     [0,0,0,0,3]
# ])

# I_N = np.identity(5)

# normalized_A = normalize_adj(A).toarray()
# normalized_L = (I_N - normalize_adj(A))
# normalized_L_trick = (I_N + normalized_A)
# # print(normalized_A)
# # print(normalized_L_trick)
# # print(normalized_L)
# adj_normalized = normalize_adj(A + sp.eye(A.shape[0]))

# # print(adj_normalized.toarray())
# # print(normalize_adj(A + I_N).toarray())

# D_new = D + I_N
# D_inv = np.power(D_new, -0.5)
# D_inv[np.isinf(D_inv)] = 0.
# print(D_inv)

adj_normalized = normalize_adj(adj) # normalize된 A (D^(-1/2)AD^(-1/2))를 계산
# print(adj_normalized)
laplacian = sp.eye(adj.shape[0]) - adj_normalized # Laplacian Matrix 계산 (L = I_N - D^(-1/2)AD^(-1/2))
largest_eigval, _ = eigsh(laplacian, 1, which='LM') # ???? --> lambda_max 계산. denotes the largest eigenvalue of L.
scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0]) # ~A를 계산

# Tk 초기 값 설정
# x=0, 1 이후엔 하위에 있는 재귀식을 통해 k차수까지 계산
t_k = list()
t_k.append(sp.eye(adj.shape[0])) # T_k(0) = 1, 현재는 행렬 형태이므로 I_N이 된다.
t_k.append(scaled_laplacian) # T_k(1) = x, 현재는 행렬 형태이므로 조정된 라플라시안 행렬인 ~A가 된다.

# 체비셰프 재귀식
# T_k(x) = 2x * T_k-1(x) - T_k-2(x) // Tk(0) = 1, Tk(1) = x
def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
    s_lap = sp.csr_matrix(scaled_lap, copy=True) # TODO: 복사?를 왜하는걸까
    return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two # T_k(x) = 2x * T_k-1(x) - T_k-2(x)

# k차수까지의 체비셰프 식 계산
# t_k 안에는 sparse matrix들이 들어있게 된다.
k=3 # Max degree of chebyshev polynomial
for i in range(2, k+1):
    t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

print(len(t_k))
# for i in range(len(t_k)):
#     print(t_k[i])
print(t_k)
# print(sparse_to_tuple(t_k[0]))