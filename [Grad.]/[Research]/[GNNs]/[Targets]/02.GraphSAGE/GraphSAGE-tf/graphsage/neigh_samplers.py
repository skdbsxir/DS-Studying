from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    # 연결된 정보 (인접행렬)을 입력으로 받아, 그 안에서 random하게 연결된 이웃을 sampling.
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        # TODO: 여기서 random shuffle!!! 들어온 샘플들을 random shuffle해서 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        # sample의 수 만큼 가져온다.
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        
        # 즉 인접행렬을 입력으로 받아서 -> embedding_lookup으로 adj_list를 만들어서 -> 이걸 random_shuffle 한 후에 -> sample의 수 만큼 (지정한 수 만큼)의 노드를 slice한다. 
        return adj_lists
