"""
CS224W

https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html#the-messagepassing-base-class
https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
"""
import numpy as np

import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


# dataset = TUDataset(root=os.getcwd() + '/datasets/ENZYMES', name='ENZYMES')
class CustomGCN(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        # aggr : MessagePassing의 aggregate function을 뭐로 할건지?
        # flow : edge 데이터가 어떻게 흐르는지? (source, target)이 있다면 보통 source_to_target으로 한다. 이게 default.
        super().__init__(aggr='mean', flow='source_to_target')

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):

        # Adj matrix에 self-loop(self-connection)을 추가. A~ = A + I
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # node feature matrix X transform.
        # node feature matrix를 linearly transform.
        x = self.linear(x)

        # propagate는 custom하게 정의 할 message 함수를 호출함. 
        # 모든 node에 대해 message를 계산하고 합쳐서 새 representation을 계산한다.
        # 인자로 담겨져있는 저것들은 무엇인가.
        # edge_index는 말 그대로 현재 어떤 노드들과 연결되어 있나? 현재 1-hop 이웃 노드가 무엇인가를 확인.
        # size는 보통 (N,N)인 square matrix인데, Bipartite 처럼 (N,M)이어도 된다함.
        # 현재는 일반적인 adj matrix이니, size는 (N,N)이 된다. 그래서 x.size(0) (# of nodes)을 2개 넣은거.
        result = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

        return result
    
    def message(self, x, edge_index, size):
        # edge_index가 뭐냐?
        # csr_matrix 처럼, 각 노드의 연결 정보를 담은 구조체. 간단히 생각해서 adj matrix라고 볼 수 있음.
        # (0, 1) (0, 2) (0, 4) (1, 2) (1, 3) (2, 4) 이런식. (source, target) : type은 보통 tensor.
        # source와 target의 쌍을 주고, 연결 정보를 보존한다. (약간 csr/coo matrix 처럼, 원소가 있는 좌표 == 연결유무 인듯?)

        # GCN의 message aggregation (recap)
        # adj matrix를 통해 이웃 정보를 가져오고, degree matrix를 이용해 Normalized Laplacian을 이용해 이웃으로부터 message를 GET.
        row, col = edge_index 
        D = pyg_utils.degree(row, x.size(0), dtype=x.dtype) # 1-dim index tensor로부터 degree를 계산
        # D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        # result = np.dot(D_inv_sqrt[row], D_inv_sqrt[col])
        result = D_inv_sqrt[row] * D_inv_sqrt[col]
        
        # 참고로, 위 과정이 전부 forward()에 들어가있음 (공식 doc tutorial에선.)
        return result.view(-1, 1) * x
    
    def update(self, aggr_out):
        # 최종 feature update 전에 무언가를 하고싶다면, 구현해서 추가할 수 있음.
        # 추가할 것이 없다면 생성 안해도 propagate가 호출해서 사용함.
        return aggr_out

# test_model = CustomGCN(32, 16)
# print(test_model.state_dict()['linear.weight'].shape)
# print(test_model.state_dict()['linear.bias'].shape)