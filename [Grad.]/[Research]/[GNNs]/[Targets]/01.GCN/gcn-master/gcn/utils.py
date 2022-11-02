import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # features = lil_matrix : row-based linked list sparse matrix
    features = sp.vstack((allx, tx)).tolil() # sp.vstack() : Stack sparse matrices vertically (row wise). np.vstack(과 동일하게. => row-wise로 쌓은 후 이를 lil matrix로 변환 -> TODO: Why?
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # adj = csr_matrix : compressed sparse row matrix => Row 순서대로 데이터를 저장.
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # train/test/val indexing
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # T/F boolean mask make with indexing
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # train/test/val slicing with mask
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# Sparse Matrix를 tuple로 변환
# COO matrix : 0이 아닌 데이터만 별도의 배열에 저장, 그 데이터가 가르키는 행과 열의 위치를 별도의 배열에 저장.
## 예제 참고 : https://leebaro.tistory.com/entry/scipysparsecoomatrix
# Sparse matrix in coordinate format.
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose() # 값이 들어있는 위치의 좌표
        values = mx.data # 해당 좌표에 들어있는 값
        shape = mx.shape # matrix의 형태
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# For experimental setup -> row-normalize input feature vectors.
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

# 인접행렬 A 정규화 : D^(-1/2)AD^(-1/2) 계산.
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # degree matrix D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# 인접행렬 A에 renormaization trick 적용
# A + I_N을 받아서 DAD 계산 -> tuple 형태로 return (값 저장 위치좌표 & 실제 값 & size)
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


# 입력을 위한 dictionary(feed_dict) 구성
# Why should we use feed_dict? https://stackoverflow.com/questions/51407644/why-we-need-to-pass-values-using-feed-dict-to-print-loss-value-in-tensorflow
# tf.placeholder를 쓰면 내부적으로 연산 그래프를 생성하고, 내부적으로 비어있는 컨테이너가 생성된다. 
# 즉 훈련 도중 변수를 가져올 빈 공간을 미리 만들어 두는 셈.
# feed_dict를 통해 빈 공간들에 변수 mapping을 한다.
# placeholder의 이점이 뭐냐? sess.run()를 호출할 때, 1번이 각각 독립적 (?) values you put in them for one execution of sess.run() are not remembered.
# 첫번째 sess.run()을 호출하고 난 후 이를 기억에서 지운다? 호출 하고 실행된 후에는 다시 빈 컨테이너가 된다.
# 단순히 연산만을 수행하는 machine으로 생각할 수 있다. 입력을 받아 실행 결과를 내뱉는.
# 이 machine은 내부적으로 값들을 저장하지 않고, 단순히 입력을 받아 출력을 내뱉는다.
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

# Paper 식(4) : 식(3)의 직접연산 문제를 해결하기 위해, g_theta를 근사화 하는 과정.
# http://dsba.korea.ac.kr/seminar/?mod=document&uid=1329 : pdf 중 ChebyNet 파트 참고
## FIXME: T_k 가 adjacency를 내포하므로, 차수를 늘려서 좀더 정교하게 연결 정보를 fix?
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

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
    # T2 부터 Tk까지 계산
    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    print(len(t_k)) # same as num_supports in train.py
    # t_k를 tuple 형태로 반환
    return sparse_to_tuple(t_k)
