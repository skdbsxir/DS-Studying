"""
- original src : https://github.com/zhulf0804/GCN.PyTorch/tree/master/misc

[load된 dataset에 대한 EDA]
- 하고자 하는 task는 document classification in citation network
    > node classification : 주어진 노드가 어느 label에 속하는가?
    > 각 document는 class label이 존재.
- 각 데이터셋은 document마다 'sparse한 BOW feature vector' & 'document 사이의 citation link' 를 포함.
    > citation link를 (undirected) edge로 간주해 그래프 구조를 binary & symmetric adj matrix로 구성
"""

import numpy as np
import os
import sys
from datasets import load_data

def get_citeseer():
    """
    train set num is 120, val set num is 500, test set num is 1000
    each class num in train set:  [20 20 20 20 20 20]
    each class num in val set:  [ 29  86 116 106  94  69]
    each class num in test set:  [ 77 182 181 231 169 160]
    """
    _, _, y_train, y_val, y_test, _, _, _ = load_data('citeseer')
    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
    train_num = np.sum(y_train)
    val_num = np.sum(y_val)
    test_num = np.sum(y_test)
    print("="*20, "citeseer", "="*20)
    print("train set num is %d, val set num is %d, test set num is %d" %(train_num, val_num, test_num))
    classes = [[] for _ in range(3)]
    for i in range(6):
        classes[0].append(np.sum(y_train[:, i]))
        classes[1].append(np.sum(y_val[:, i]))
        classes[2].append(np.sum(y_test[:, i]))
    types = ['train set', 'val set', 'test set']
    classes = np.array(classes, dtype=np.int)
    for i in range(3):
        print("each class num in %s: "%types[i], classes[i])


def get_cora():
    """
    train set num is 140, val set num is 500, test set num is 1000
    each class num in train set:  [20 20 20 20 20 20 20]
    each class num in val set:  [ 61  36  78 158  81  57  29]
    each class num in test set:  [130  91 144 319 149 103  64]
    """
    _, _, y_train, y_val, y_test, _, _, _ = load_data('cora')
    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
    train_num = np.sum(y_train)
    val_num = np.sum(y_val)
    test_num = np.sum(y_test)
    print("=" * 20, "citeseer", "=" * 20)
    print("train set num is %d, val set num is %d, test set num is %d" % (train_num, val_num, test_num))
    classes = [[] for _ in range(3)]
    for i in range(7):
        classes[0].append(np.sum(y_train[:, i]))
        classes[1].append(np.sum(y_val[:, i]))
        classes[2].append(np.sum(y_test[:, i]))
    types = ['train set', 'val set', 'test set']
    classes = np.array(classes, dtype=np.int)
    for i in range(3):
        print("each class num in %s: " % types[i], classes[i])


def get_pubmed():
    """
    train set num is 60, val set num is 500, test set num is 1000
    each class num in train set:  [20 20 20]
    each class num in val set:  [ 98 194 208]
    each class num in test set:  [180 413 407]
    """
    _, _, y_train, y_val, y_test, _, _, _ = load_data('pubmed')
    y_train, y_val, y_test = y_train.numpy(), y_val.numpy(), y_test.numpy()
    train_num = np.sum(y_train)
    val_num = np.sum(y_val)
    test_num = np.sum(y_test)
    print("=" * 20, "citeseer", "=" * 20)
    print("train set num is %d, val set num is %d, test set num is %d" % (train_num, val_num, test_num))
    classes = [[] for _ in range(3)]
    for i in range(3):
        classes[0].append(np.sum(y_train[:, i]))
        classes[1].append(np.sum(y_val[:, i]))
        classes[2].append(np.sum(y_test[:, i]))
    types = ['train set', 'val set', 'test set']
    classes = np.array(classes, dtype=np.int)
    for i in range(3):
        print("each class num in %s: " % types[i], classes[i])

get_citeseer()
get_cora()
get_pubmed()