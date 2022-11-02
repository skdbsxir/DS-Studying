from collections import namedtuple

import tensorflow as tf
import math

import graphsage.layers as layers
import graphsage.metrics as metrics

from .prediction import BipartiteEdgePredLayer
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# For testing
import sys
sess = tf.compat.v1.Session()

# GCN과 동일
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

# Loss 구분 이외엔 GCN과 동일
class MLP(Model):
    """ A standard multi-layer perceptron """
    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                 output_dim=self.dims[1],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 dropout=self.placeholders['dropout'],
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

# 전통적인 layer로 구축할 수 없는 모델을 구축하기 위한 class
class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        # 계산그래프에서 정의된 variable collection을 저장 (for easy access)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
 ## 반복되는(recursive한) GraphSAGE 층을 명시(구분)하기 위한 namedtuple
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", identity_dim=0,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        '''
            - placeholders : 필요한 정보들을 담고있는 저장체? 통? 이라고 생각하면 됨
            - features : node feature (ndarray) 
            - adj : 노드의 연결 정보가 담긴 list로 구성된 ndarray -> 인접행렬 -> A
            - degrees : 노드의 차수 정보를 담고있는 ndarray -> D
            - layer_infos : 앞서 정의한 SAGEInfo -> 반복되는 GraphSAGE층을 구분하기 위해 layer간의 정보를 담은 구조체
            - concat : recursive iteration을 하는 동안 concat을 할지, 말지? default=True
            - aggregator_type : 자기 자신과 이웃의 representation을 집계할 layer 안의 aggregatory type 정의
            - identity_dim : 항등 feature를 사용 여부 default=0
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)

        # Agg. type 정의
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
         ## Batch 단위로 수행되므로, placeholder의 batch들을 input으로 받는다.
         ### FIXME: 
          ## batch에는 random sampling된 input feature들이 들어가있다?
          ## 각 batch B^{k}는 다음 batch B^{k+1}에 포함된 노드 v의 representation을 계산하기 위한 node들이 포함되어 있음.
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.adj_info = adj

        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        
        # train in featureless mode (identity features for nodes)
        ## FIXME: 노드에 기본적으로 feature가 없다고 가정하고 항등 feature로 적용
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        ## 노드에 feature가 있으면 그대로 지정
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)

        self.degrees = degrees
        self.concat = concat

        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        # Optimizer (FIXME: paper는 minibatch를 위해 SGD를 쓴다고 하지 않았던가..?)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        """
        연산을 위해 주변 이웃들 중 일부를 sampling
        """
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # print(inputs) # Tensor("batch1:0", dtype=int32)
        # print(tf.print(inputs, output_stream=sys.stderr))
        # print(type(inputs)) # <class 'tensorflow.python.framework.ops.Tensor'>


        # Batch로 들어온 내용(자신과 이웃 노드의 representation 정보들) -> sample
        # [<tf.Tensor 'batch1:0' shape=<unknown> dtype=int32>]
        samples = [inputs]

        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]

        # 서로 다른 layer 수 만큼 loop를 반복
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1 # flag

            # 현재의 support size와 해당 layer에서의 sample수를 곱해 support size를 재 정의
            ## FIXME: support? GCN에선 K차수(hidden layer 갯수)로 이해. 여기도 같은의미?
            support_size *= layer_infos[t].num_samples

            # print('Support Size is')
            # print(support_size) # 10 (with toy_ppi)
            # print('#############'*8 + '\n')

            # sampler 구조 정의 -> batch 안에서 이웃을 어떻게 sample 할 것인지?
            ## 추후 XXX_train.py에서 neigh_samplers.py를 import 한 후, layer를 설계할때 layer_info에 전달
            sampler = layer_infos[t].neigh_sampler

            # sampling된 이웃 노드
            node = sampler((samples[k], layer_infos[t].num_samples))
            
            # print('Values in node')
            # # tf.print(node, output_stream=sys.stderr)
            # print(node) 
            # # print(tf.keras.backend.get_value(node)) # feed_dict를 해줘야..
            # print('#############'*8 + '\n')

            # FIXME: sampling된 이웃 노드를 sample에 새로 추가?
            ## (neigh_samplers.py : Assumes that adj lists are padded with random re-sampling)
            ## re-sample된 정보들을 다시 추가해, 집계과정에서 focusing?
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)

            # print('Samples is ')
            # print(samples) # [<tf.Tensor 'batch1:0' shape=<unknown> dtype=int32>, <tf.Tensor 'Reshape:0' shape=(?,) dtype=int32>, <tf.Tensor 'Reshape_1:0' shape=(?,) dtype=int32>]
            # tf.print(samples, output_stream=sys.stderr) # [tensor, tensor, tensor]
            # print('\nSupport Size is ')
            # print(support_sizes) # [1, 10, 250]
            # print('#############'*8 + '\n')

        return samples, support_sizes


    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        """
        각 layer마다 aggregate 수행 -> 이웃의 hidden representation을 집계, next layer의 hidden representation을 계산 (h^{k-1}들을 가지고 h^{k}를 계산)

            - samples : 각 layer에서 conv를 위해 다양한 hop만큼 떨어진 sample들의 집합 => conv를 위해 hop=1,2,... 만큼 떨어진 이웃 노드들의 집합(list)
                        len(samples) = # of layers + 1
                        각각의 원소는 node index를 의미하는 vector
            - input_features : 여러 hop만클 떨어진 sample 각각의 feature (initial representation)
            - support_sizes : 각 layer로부터 정보를 수집할 노드의 수 # FIXME: 선택할 이웃의 수?

        Return은 batch안의 모든 노드들에 대한 마지막 layer에서의 hidden representation -> h^{k}
        """
        # print(tf.Session().run(samples[0]))
        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        # layer마다 정의한 aggregator 정의 -> for train aggregator by loss_fn
        for layer in range(len(num_samples)):
            # Aggregator 정의
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                # 현재 layer에서의 aggregator 
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                # 다른 layer에서의 aggregator?
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size) 
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]

            # hidden representation at current layer for all support nodes that are various hops away
            # TODO: 현재 layer에서의 hidden representation (hv^{k}) : 다양한 hop만큼 떨어진 모든 주변 노드에서 정보를 받아와야 한다 (?)
            next_hidden = []
            
            # as layer increases, the number of support nodes needed decreases
            # TODO: layer가 증가하면, 주변 노드(support node)의 수는 감소해야 한다?
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        
        # fixed base distribution에 따라 negative sampling 수행
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75, # 왜도 정도 정의? -> (doc) skew the unigram probability distribution.
            unigrams=self.degrees.tolist()))

           
        ### perform "convolution"
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        # 노드 자신의 집계된 정보 and 이웃 노드들의 집계된 representation
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs2, _ = self.aggregate(samples2, [self.features], self.dims, num_samples,
                support_sizes2, aggregators=self.aggregators, concat=self.concat,
                model_size=self.model_size)

        # random walk 시퀀스 안에서의 negative sampling
        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos,
            FLAGS.neg_sample_size)

        # negative sample (멀리 떨어진 노드 -> 지정된 hop보다 멀리 떨어진 노드)에서 집계된 representation
        self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                concat=self.concat, model_size=self.model_size)
        ###

        dim_mult = 2 if self.concat else 1

        # TODO: link predict 수행? -> accuracy 계산을 위해 BipartiteEdgePredLayer로 전달 (하단 _accuracy 참고)
        self.link_pred_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid, 
                bilinear_weights=False,
                name='edge_predict')

        # Normalize output
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        # Aggregator의 loss 계산 
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs) 
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2) # https://github.com/williamleif/GraphSAGE/issues/95 (why use link_pred_layer)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


class Node2VecModel(GeneralizedModel):
    def __init__(self, placeholders, dict_size, degrees, name=None,
                 nodevec_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Node2VecModel, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.degrees = degrees
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.batch_size = placeholders['batch_size']
        self.hidden_dim = nodevec_dim

        # following the tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
                tf.random_uniform([dict_size, nodevec_dim], -1, 1),
                name="target_embeds")
        self.context_embeds = tf.Variable(
                tf.truncated_normal([dict_size, nodevec_dim],
                stddev=1.0 / math.sqrt(nodevec_dim)),
                name="context_embeds")
        self.context_bias = tf.Variable(
                tf.zeros([dict_size]),
                name="context_bias")

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])

        # negative sampling -> Samples a set of classes using the provided (fixed) base distribution. (fixed_unigram_candidate_sampler)
        # 최대 차수만큼의 범위 안에서 random하게 class를 sample.
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        self.outputs1 = tf.nn.embedding_lookup(self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(self.context_embeds, self.inputs2)
        self.outputs2_bias = tf.nn.embedding_lookup(self.context_bias, self.inputs2)

        # neg sample -> embedding
        self.neg_outputs = tf.nn.embedding_lookup(self.context_embeds, self.neg_samples)
        self.neg_outputs_bias = tf.nn.embedding_lookup(self.context_bias, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.hidden_dim, self.hidden_dim,
                self.placeholders, bilinear_weights=False)

    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._minimize()
        self._accuracy()

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(self.outputs1, self.outputs2), 1) + self.outputs2_bias
        neg_aff = tf.matmul(self.outputs1, tf.transpose(self.neg_outputs)) + self.neg_outputs_bias
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        tf.summary.scalar('loss', self.loss)
        
    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
       # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
