from __future__ import division
from __future__ import print_function

from graphsage.inits import zeros
from graphsage.layers import Layer
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Bipartite Edge -> 이분 그래프의 edge (서로의 연결이 의미가 있는 edge들?)
class BipartiteEdgePredLayer(Layer):
    def __init__(self, input_dim1, input_dim2, placeholders, dropout=False, act=tf.nn.sigmoid,
            loss_fn='xent', neg_sample_weights=1.0,
            bias=False, bilinear_weights=False, **kwargs):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be 
                based on dot product.
        """
        """
        Skip-gram은 중심 단어를 기준으로 주변 단어를 예측하는 알고리즘
         -> skip-gram-like loss : metric learning's objective
        """
        super(BipartiteEdgePredLayer, self).__init__(**kwargs)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias
        self.eps = 1e-7

        # Margin for hinge loss
         ## The hinge loss is a specific type of cost function that incorporates a margin or distance from the classification boundary into the cost calculation. 
         ## Even if new observations are classified correctly, they can incur a penalty if the margin from the decision boundary is not large enough. 
        self.margin = 0.1 # negative 간의 margin 값 (차이가 얼마나 나는지에 대한 임계값) -> max-margin objective

        ### FIXME: ??
        # negative sample에 적용할 weight
        self.neg_sample_weights = neg_sample_weights
        # 이게 뭘까 -> Zu, Zvn이 같이 있으니, 이를 tuning하기 위한 weight?
        self.bilinear_weights = bilinear_weights
        ### FIXME:

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        # output a likelihood term
        self.output_dim = 1
        with tf.variable_scope(self.name + '_vars'):
            # bilinear form
            ## 말 그대로 이중선형(bilinear) -> Zu * Zvn 이렇게 붙어있으니까?
            if bilinear_weights:
                #self.vars['weights'] = glorot([input_dim1, input_dim2],
                #                              name='pred_weights')
                
                ### FIXME: pred_weight? 
                # tuning에 활용할 weight 새롭게 정의 & initialize
                self.vars['weights'] = tf.get_variable(
                        'pred_weights', 
                        shape=(input_dim1, input_dim2),
                        dtype=tf.float32, 
                        initializer=tf.contrib.layers.xavier_initializer())

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        # Define loss functions
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        elif loss_fn == 'skipgram':
            self.loss_fn = self._skipgram_loss
        elif loss_fn == 'hinge':
            self.loss_fn = self._hinge_loss

        if self.logging:
            self._log_vars()

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        """
        식(1)에서 앞 부분 파트. -log(sigma(zu^T * zv))
         -> nearby nodes to have similar representations
         -> 현재 자신의 representation과 근처 이웃노드(positive sample/node)간의 affinity score는 높아지는 방향으로
        """

        # 두개의 입력 input1과 input2의 affinity score(NOTE:관련성 점수 -> 결국 similarity를 의미) 계산
        # shape: [batch_size, input_dim1]
        ## TODO: 이 if-else 구분이 정확이 뭘 위한것일까. bilinear_weight?
        if self.bilinear_weights:
            prod = tf.matmul(inputs2, tf.transpose(self.vars['weights'])) # WT * 
            self.prod = prod
            result = tf.reduce_sum(inputs1 * prod, axis=1)
        else:
            result = tf.reduce_sum(inputs1 * inputs2, axis=1)
        ## 
        return result

    def neg_cost(self, inputs1, neg_samples, hard_neg_samples=None):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        """
        식(1)에서 뒷 부분 파트. -Q*Elog(sigma(-zu^T * zv)) 
         -> representation of disparate nodes are highly distinct
         -> 현재 자신의 representation과 떨어진 이웃노드(negative sample/node)간의 affinity score는 낮아지는 방향으로 
        """
        if self.bilinear_weights:
            inputs1 = tf.matmul(inputs1, self.vars['weights'])
        neg_aff = tf.matmul(inputs1, tf.transpose(neg_samples))
        return neg_aff

    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        # SGD를 위해 하단에 제시된 loss함수에 따라 loss 계산
        return self.loss_fn(inputs1, inputs2, neg_samples)


    """
    Paper에서 제안한 loss함수들 -> unsupervised loss functions (to be trained without task-specific supervision)
    """
    # xent loss는 paper에서 제시한 loss function -> 식(1)
    # https://github.com/williamleif/GraphSAGE/issues/95 (why use link_pred_layer)
    # https://github.com/williamleif/GraphSAGE/issues/14 (some explanation)
     ## The loss in the original paper is equivalent to _xent_loss in the code, 
     ## which is a typical word2vec skip-gram negative-sampling loss which is roughly equivalent to nce loss.
        ### NCE(Noise Contrastive Estimation) : https://nuguziii.github.io/survey/S-006/ (well-explained) // https://89douner.tistory.com/334 (Contrastive Learning)
     ## https://www.baeldung.com/cs/nlps-word2vec-negative-sampling (식(8) : skip-gram with negative-sampling loss)
    def _xent_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):

        # 자신과 근처 이웃의 affinity score 계산 -> zu^T * zv
        aff = self.affinity(inputs1, inputs2) 

        # 자신과 멀리 떨어진 이웃(negative sample)의 affinity score 계산 -> -zu^T * zv
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)

        # 위에서 구한 각각의 affinity score에 대해 sigmoid를 씌운 실제 값으로 계산
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)

        # (axis가 명시되지 않은 reduce_sum은 모든 요소를 sum해서 단일 tensor(single value)로 반환.)
        # 실제 식(1) 부분.
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss

    # skip-gram like loss
    def _skipgram_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):

        # 마찬가지로 자신과 근처 이웃의 affinity score 계산
        aff = self.affinity(inputs1, inputs2)

        # 마찬가지로 자신과 멀리 떨어진 이웃(negative sample)의 affinity score 계산
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)

        # TODO: ??
         ## negative sample과의 affinity score에 exp를 취한 후 reduce_sum : axis=1로 모든 성분의 합 계산
         ## 이 결과에 log를 취해 다시 변환 ??
        neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))

        # 근처 이웃과의 affinity score와 최종 변환계산된 negative sample과의 affinity score를 뺀 값의 reduce_sum 계산
        loss = tf.reduce_sum(aff - neg_cost)
        return loss

    # Hinge Loss
     ## https://medium.com/analytics-vidhya/understanding-loss-functions-hinge-loss-a0ff112b40a1 
     ## https://ai.stackexchange.com/questions/26330/what-is-the-definition-of-the-hinge-loss-function
    def _hinge_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        # 이 과정은 위와 동일
        aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        
        # TODO: 실제와 negative간의 margin 정의 -> 이를 줄이도록?
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 1) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        self.neg_shape = tf.shape(neg_aff)
        return loss

    # Normalizer (Unused)
    def weights_norm(self):
        return tf.nn.l2_norm(self.vars['weights'])
