"""
GCN layer 정의
"""
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.parameter as Parameter
import torch.nn.functional as F
from datasets import load_data

def normalize_adj(adj:np.array):
    """Symmetrically normalize adjacency matrix"""
    rowsum = np.array(adj.sum(1)) # degree matrix D의 구성을 위해 adj의 각 행별로 sum 수행.
    D_inv_sqrt = np.power(rowsum, -0.5) # D^(-1/2) 계산 (이는 현재 rowsum을 수행했으므로, 1차원 vector 형태.)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0. # 위 연산에서 inf인 부분은 0으로 replace.
    D_inv_sqrt = np.diag(D_inv_sqrt) # 1차원 vector형태인 D^(-1/2)를 diagonal matrix로 변환 --> 구하고자 하는 최종 degree matrix 가 된다.
    result = np.dot(D_inv_sqrt, np.dot(adj, D_inv_sqrt)) # 구하고자 하는 최종 DAD

    return result

def preprocess_adj(adj:np.array):
    """Preprocess adj by adding self-connection"""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0])) # Identity matrix I_N(np.eye(adj.shape[0]))을 더해 self-connection을 추가.

    # print(adj_normalized.shape)
    # # print(np.count_nonzero(adj_normalized))
    # sparsity = 1. - (np.count_nonzero(adj_normalized) / float(adj_normalized.size))
    # print(sparsity) # (cora) adj matrix sparsity : 0.9981

    return adj_normalized

# adj, _, _, _, _, _, _, _ = load_data('cora')
# preprocess_adj(adj)

class GraphConvolution(nn.Module):
    """Simple GCN layer"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None) # bias=False인 경우 bias 등록
        
        self.init_weights() # weight & bias 초기화.

    def init_weights(self):
        bound = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)
    
    def forward(self, input, adj):
        """layer-wise propagation rule (Z=DAD*X*W)"""

        XW = torch.mm(input, self.weight) # X*W
        Z = torch.spmm(adj, XW) # DAD * XW

        """
        # TODO: 원저자의 위 연산과 아래 연산의 차이가 있을까?
        # TODO: 시간을 비교해보면 될듯. time.time()
            ## 근데 느낌상 W의 col (# of out_feature) 수가 작으니 XW가 빠를 것 같긴 하다.
        DADX = torch.spmm(adj, input)
        Z = torch.spmm(DADX, self.weight)
        """
        
        if self.bias is not None:
            return Z + self.bias
        else:
            return Z
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'