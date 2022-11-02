"""
GCN layer 정의
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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