"""
Utility functions
"""
import torch
import torch.nn as nn
import numpy as np


class Loss(nn.Module):
    """Calculate loss"""
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    # original repo는 nll loss 를 사용
    # data loading이 original repo와 다르므로 재계산
    def forward(self, output, labels, mask):
        labels = torch.argmax(labels, dim=1)
        loss = self.loss(output, labels)
        mask = mask.float()
        mask /= torch.mean(mask)
        loss *= mask
        
        return torch.mean(loss)

def calc_loss(output, labels, mask):
    loss = Loss()
    return loss(output, labels, mask)

def accuracy(output, labels, mask):
    """Calculate accuracy by label and model's output"""
    
    output = torch.argmax(output, dim=1).numpy()
    labels = torch.argmax(labels, dim=1).numpy()
    correct = output == labels

    # print(correct)
    
    # masking accuracy
    mask = mask.float().numpy()
    TP = np.sum(correct * mask)

    acc = TP / np.sum(mask)

    return acc

def sparse_matrix_to_torch_sparse_tensor(sparse_matrix):
    """Convert sparse matrix (ndarray) to torch sparse tensor."""

    # adj는 현재 scipy의 csr_matrix
    # coo_matrix로 변환 후 sparse tensor로 변환
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64))
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)