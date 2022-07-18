import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros

"""
GCN -> .layers 내부에서 GraphConvolution class가 같이 정의.
GraphSAGE -> .layers의 Layer, Dense를 상속받아 Aggregator를 정의
            => Neural network layer이므로, trainable하다.
"""

class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat # 자기자신과 이웃의 concat 여부 (FIXME: 이웃을 선택하지 않는다면 자기자신과 자기자신?)

        # 이웃 정보를 받지 않는다면? -> 자기 자신만을 입력으로 받는다.
        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        # Aggregator layer의 내부 weight W 초기화.
        # weight는 2가지 : 이웃 노드에 적용될 neigh_weights // 자기 자신 노드에 적용될 self_weights
        # FIXME: paper에서 이에 대한 언급이 있었나..? 
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        # 단순히 h^(k-1)의 모든 벡터에 대해 element-wise mean을 계산
        # 기존 (transductive) GCN propagation rule과 동일한 방법
        
        # Aggregator layer의 input은
        # 1) 이전 layer에서의 자기 자신의 representation hv^{k-1}
        # 2) 이전 layer에서의 주변 이웃의 representation hu^{k-1}
        self_vecs, neigh_vecs = inputs

        # dropout 적용
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # 주변 이웃들의 representation의 평균 계산 (by axis=1 : 행 방향 => tensor의 왼쪽에서 오른쪽 방향으로)
        # 주변 이웃들의 정보(representation)를 우선 집계
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1) # 이웃 노드들의 평균을 먼저 계산 (hu^{k-1}) 
       
        # [nodes] x [out_dim]
        # {k-1}때의 주변 이웃과 자기 자신의 정보(representation) * (learnable) Weight
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights']) # W * hu^{k-1}

        from_self = tf.matmul(self_vecs, self.vars["self_weights"]) # W * hv^{k-1}
         
        if not self.concat:
            # add_n : tf_add와 동일하지만, 많은 양의 텐서를 한번에 처리할 수 있음.
            # https://www.tensorflow.org/api_docs/python/tf/math/add_n
            output = tf.add_n([from_self, from_neighs])
        else:
            # FIXME: concat(axis=1) : 단순히 오른쪽으로 붙이기. sum? GCN prop. rule?
            # GCN.layers 안의 GraphConvolution class를 보면 dot 후에 append 한다. propagation rule이 맞아 보인다.
            output = tf.concat([from_self, from_neighs], axis=1) # 이 둘을 단순히 concat -> transductive GCN propagation rule (matmul --> sum)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat # 마찬가지로 자기자신과 이웃의 concat 여부 (마찬가지로 FIXME: 이웃을 선택하지 않는다면 자기자신과 자기자신?)

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 기존 GCN과 동일하게 1개의 weight만을 사용.
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        # 기존 transductive한 GCN의 propagation rule을 inductive 한 방법으로 변경
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)

        # MEAN(hv^{k-1} UNION hu^{k-1}) (식(2))
        # tf.expand_dims => 해당하는 axis를 확장
        # https://www.tensorflow.org/guide/tensor : 생김새 헷갈릴때 참고
        
        # NOTE: neigh_vecs는 이웃의 수에 따라 형태가 달라지므로(?), self_vecs의 차원을 확장한 후 concat 수행
        # (testing.py) (2,1,2), (2,2) -> (2,1,2), (2,1,2) =(concat)=> (2,2,2) : 이웃들의 representation과 자기 자신 representation의 mean 계산 
        # concat : 각기 다른 탐색 깊이(layer)간의 skip-connection의 일종으로 간주 가능
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        # 최종 결과물은 mean에 weight를 곱한 값.
        output = tf.matmul(means, self.vars['weights']) # W * MEAN(~~) -> activation을 거치면 최종 output = hv^k 가 생성된다.

        # bias
        if self.bias:
            output += self.vars['bias']

        # Final activation : hv^{k}, 즉 현재의 hidden state.
        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    # PointNet에서 영감을 받은 구조
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        # Define MLP(pooling) layer's input size
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        # Element-wise max pooling dense layer. (여기선 1-layer만 생각)
        # MLP는 이웃 집합(set)에 있는 node representation 각각을 계산하는 function의 집합으로 생각할 수 있음.
        ## Pooling을 통해 이웃 node의 각기 다른 관점에서의 feature를 포착할 수 있음.
        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        # Weight Initialize (self weight, neighbor weight)
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        # max-pooling with 1 layer -> 식(3)
        # FIXME: self_vecs는 단일 tensor가 될 것이고, neight_vecs는 단일 tensor가 집계한 이웃의 수 만큼 묶여있는 1차원 더 높은 tensor가 될 것.
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs # 이웃의 hidden states -> 아웃 수 만큼의 내용을 가진 tensor : [[이웃1_1, 이웃1_2], [이웃2_1 이웃2_2], ... , (이웃n_1, 이웃n_2)] : shape는 (feature크기, 이웃갯수)가 될 것.

        dims = tf.shape(neigh_h) # neigh_h의 shape. (feature크기, 이웃갯수)
        batch_size = dims[0] # feature 크기 : FIXME:현재 minibatch로 찢었으므로 크기가 각 batch size가 된다.
        num_neighbors = dims[1] # 이웃의 수

        # [nodes * sampled neighbors] x [hidden_dim]
        # matmul을 위한 reshaping => FIXME:symmetric하므로 reshape해도 문제가 없다? => Dense(pooling) layer에서 받아들일 입력 크기에 맞춰 reshape
        ## neight_h의 크기는 (feature크기(batch_size), 이웃갯수) => (batch_size*이웃갯수, Dense입력크기)
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # Dense layer를 거쳐 이웃의 h를 계산 -> Dense class의 _call()이 수행. (Dense의 weight * 입력 h)
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)

        # reshaping헀던 h를 다시 원래 차원으로 복원
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        
        # NOTE: max-pooling
        # tensor의 각 차원에서 max-pooling : 가장 큰 요소들을 뽑아 이웃의 representation으로 지정
        # pooling은 h에서 각 batch마다 가로 방향('>'방향)으로 max-pooling : 각 요소들의 max를 골라 새 요소 생성
        # neighbor aggregation with max-pooling
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        # NOTE: 여기까지가 식(3) -> AGGREGATE 함수. -> Algo_1, line_4 : 이웃 aggregation (hN(v)^{k} 계산)

        
        # 집계된 이웃 rep.와 자신의 rep.에 weight를 곱한 후 activation -> 최종 hv^{k}
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    # 앞 max-pooling과 동일한 구조이지만, pooling에서 max가 아닌 mean 방법을 적용.
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 마찬가지로 MLP(pooling) layer's input size 정의
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        # 마찬가지로 Element-wise mean pooling dense layer
        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        # Weight Initialize (self & neighbor)
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        # mean-pooling with 1 layer -> 식(3)에서 max가 아닌 mean
        # max-pooling과 동일한 전처리작업
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        # [nodes * sampled neighbors] x [hidden_dim]
        # 마찬가지로 matmul을 위한 reshaping
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # Dense layer를 거쳐 이웃의 h를 계산
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        
        # reshaping했던 h를 다시 원래 차원으로 복원
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))

        # NOTE: mean-pooling
        # tensor의 각 차원에서 mean-pooling : 요소 각각의 평균을 계산해 이웃의 representation으로 지정
        # 마찬가지로 각 batch마다 가로 방향 ('>'방향)으로 mean-pooling : 각 요소들의 평균을 계산해 새 요소 생성
        # neighbor aggregation with mean-pooling
        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        # NOTE : 여기까지가 식(3)(에서 max->mean) -> AGGREGATE 함수. -> Algo_1, line_4 : 이웃 aggregation (hN(v)^{k} 계산)


        # 마찬가지로 집계된 이웃 rep.와 자신의 rep.에 weight를 곱한 후 activation -> 최종 hv^{k}
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    # 앞 2개의 pooling과 동일하지만, 여기는 Dense layer가 2개. 즉 Pooling작업을 2번 한다.
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        # Pooling연산을 2번 -> Dense(Pooling) layer가 2개
        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim_1,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                 output_dim=hidden_dim_2,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        # Weight Initialize
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        # 층이 2개인 maxpooling agg. -> 식(3)에서 sigmoid가 2개
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        # 2개의 Dense layer를 거쳐 이웃의 h를 계산 (FIXME: Wh+b만 2번계산한 후에 max?)
        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)

        # 2개 layer를 거쳐서 계산된 h에서 max-pooling해 neighbor aggregation
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(neigh_h, axis=1) # max-pooling
        
        # 2개 layer를 거쳐 집계된 이웃 rep.와 자신의 rep.에 weight를 곱한 후 activation -> 최종 hv^{k}
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    # LSTM 구조에 기반한 Aggregator
    # https://github.com/williamleif/GraphSAGE/issues/20 : why lstm agg. obtain highest f1 score?
    ## LSTMs can in principle learn to approximate any permutation-invariant function
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

        # 기본 LSTM cell 구성
        # state size = hidden_dim(128 or 256)
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        # LSTM을 이웃 노드의 random permutation에 적용 -> unordered set에서도 동작할 수 있도록.
        # LSTM은 기본적으로 symmetric하지 않음 -> 입력을 순차적으로 처리 : not permutation invariant (입력에 따라 출력이 달라질 수 있음)
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]

        # float32 type으로 내부 hidden state를 0. 으로 초기화
        initial_state = self.cell.zero_state(batch_size, tf.float32)

        ### FIXME: 이 파트들이 왜 수행되는걸까?
        # FIXME: 이웃 벡터 크기를 (max-pooling같은 방식으로)줄이고, 여기에 sign(부호)을 매긴다?
        # TODO: abs를 취해버리면 전부 양수가 되는데, 여기서 sign을 매기면 전부 1이 되어버려서 전부 사용한다는 의미?
        # https://github.com/williamleif/GraphSAGE/issues/69 : 참고
         ## Here we are defining length by the size of the (sampled) neighborhood.
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))

        # axis=1 방향으로 summation -> 함축된 tensor
        length = tf.reduce_sum(used, axis=1)

        # 축소된 tensor와 1과 크기를 비교 -> FIXME: 근데 abs로 넘어오고 전부 1보다 큰데, 그러면 length가 무조건 return 되는 것 아닌가? 
        length = tf.maximum(length, tf.constant(1.))
        
        # float -> int 형변환
        length = tf.cast(length, tf.int32)
        ### 

        with tf.variable_scope(self.name) as scope:
            try:
                # dynamic rnn? https://stackoverflow.com/questions/43100981/what-is-a-dynamic-rnn-in-tensorflow 
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)

        # Random permutation?
        # rnn_outputs : [batch_size, max_len, output_size]의 shape을 가진 tensor.
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])

        ### FIXME: 보충필요
        # 임의의 sequence of number 생성
        index = tf.range(0, batch_size) * max_len + (length - 1)

        # tensor flatten to 1d
        flat = tf.reshape(rnn_outputs, [-1, out_size])

        # FIXME: flatten된 tensor에서 index에 따라 임의로 sampling? 
        # https://www.tensorflow.org/api_docs/python/tf/gather : ???
        # 단순히 index에 맞춰서 random permutation을 수행, 이웃의 representation을 정의.
        neigh_h = tf.gather(flat, index)
         ## 이웃의 repreentation.
         ## https://github.com/williamleif/GraphSAGE/issues/69 : 비슷한 질문
        ###


        # 집계된 이웃 rep.와 자신의 rep.에 weight를 곱함.
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        # 이 둘을 element-wise하게 덧셈.
        output = tf.add_n([from_self, from_neighs])

        # TODO: 이건 뭐지? 바로 위에서 add_n을 수행하는데?
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

