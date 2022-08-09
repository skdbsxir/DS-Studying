import networkx as nx
import numpy as np
import os
import pickle # to load datasets
import torch

import warnings

warnings.filterwarnings('ignore')

# For experimental setup -> row-normalize input feature vectors.
def normalize_feature(features):
    """Normalize input features."""
    row_sum_diag = np.sum(features, axis=1) # row 방향으로 summation -> Degree matrix D
    row_sum_diag_inv = np.power(row_sum_diag, -1) # Degree matrix D의 inverse matrix -> D^(-1)
    row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0. # 무한대로 기록된 값은 0으로 대체
    row_sum_inv = np.diag(row_sum_diag_inv)
    features = row_sum_inv.dot(features)

    return features

# Masking
    ## Adj에서 일부를 masking
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# Kipf's PyTorch implementation -> only cora
# original src : https://github.com/zhulf0804/GCN.PyTorch/blob/master/datasets.py
    ## shape에 대한 comment는 모두 cora 기준. 
    ## Cora : 2708 nodes, 5429 edges, 7 classes, 1433 features
def load_data(dataset_str):
    """Load dataset from 'data' folder."""
    print(f'Loading {dataset_str} dataset...')

    data_path = './data/'
    suffixes = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph'] # 파일 명 끝 부분들. (.x, .allx, .graph, ... -> pickled numpy files)
    objects = []

    for suffix in suffixes:
        file = os.path.join(data_path, f'ind.{dataset_str}.{suffix}') # 데이터셋에 해당하는 모든 파일 load
        objects.append(pickle.load(open(file, 'rb'), encoding='latin1')) # 원본 파일이 pickle로 압축된 파일이므로 pickle.load를 통해 load
    x, y, allx, ally, tx, ty, graph = objects # load한 파일(object)들을 각 변수명에 할당 -> 원본처럼 tuple로 읽던, 그냥 objects로 읽던 차이가 없어보인다.

    # for i in range(len(objects)):
    #     print(type(objects[i]))#.__name__)
    """
        x : scipy.sparse.csr.csr_matrix (csr_matrix : compressed sparse row matrix => Row 순서대로 데이터를 저장.)
            (DeprecationWarning: Please use `csr_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.csr` namespace is deprecated.)
        y : numpy.ndarray
        allx : scipy.sparse.csr.csr_matrix
        ally : numpy.ndarray
        tx : scipy.sparse.csr.csr_matrix
        ty : numpy.ndarray
        graph : collections.defaultdict 
            > cora : 2708 nodes // citeseer : 3327 nodes // pubmed : 19717 nodes
    """

    x, allx, tx = x.toarray(), allx.toarray(), tx.toarray() # csr_matrix -> ndarray 변환


    # Test dataset indexing
        ## 단순히 test때 사용할 node의 index를 가진 파일.
        ## 원본에서의 parse_index_file 함수 part
    test_index_file = os.path.join(data_path, f'ind.{dataset_str}.test.index')
    with open(test_index_file, 'r') as f:
        lines = f.readlines()
    indices = [int(line.strip()) for line in lines]
    min_index, max_index = min(indices), max(indices) # cora (1708, 2707) // citeseer (2312, 3326) // pubmed (18717, 19716)


    # Preprocess test indices and combine all data
        ## 원본에서의 if dataset == 'citeseer' 부분(과는 살짝 다르다)
        ## 고립된 노드를 처리하기 위해, 이들을 찾아 zero-vec처럼 더해 올바른 위치에 둔다.
        ## Find isolated nodes, add them as zero-vecs into the right position
    tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1])) # (1000, 1433)
    features = np.vstack([allx, tx_extend]) # (2708, 1433) -> feature matrix X : (전체 노드의 수, 전체 feature의 수)
    features[indices] = tx

    ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
    labels = np.vstack([ally, ty_extend])
    labels[indices] = ty

    
    # Build adjacancy matrix A (ndarray)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).toarray()
    

    # train-val-test splitting & masking
    idx_train = range(len(y)) # 140 (0~140)
    idx_val = range(len(y), len(y) + 500) # 500 (140~640)
    idx_test = indices # 1000 (test.index 파일)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]


    # Transform data as torch tensors
    features = torch.from_numpy(normalize_feature(features))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
        torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)
    # print(features.size())
    # print(np.count_nonzero(features))
    # sparsity = 1. - (np.count_nonzero(features) / float(np.array(features).size))
    # print(sparsity) # (cora) feature matrix sparsity : 0.9873

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# load_data('cora')