from __future__ import division
from __future__ import print_function

import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

tf.logging.set_verbosity(tf.logging.ERROR)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# Default data is 'cora'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# print(adj) # 1이 기록된 위치 (row, col)를 가지고 있다.
# print(type(adj)) # csr_matrix
# print(adj.shape) # (2708, 2708)
# print(adj.shape[0])
# print(adj.getnnz()) # 10556

# print(features) # adj처럼 기록된 위치 (row, col)을 가지고 있지만, type이 lil_matrix

# print(y_train.shape) # sparse binary ndarray. (2708, 7)
# print(train_mask.shape) # boolean ndarray. (2708,)

features = preprocess_features(features)
print(features) # tuple => (ndarray, ndarray, tuple)
print(type(features[0])) 
print(type(features[1]))
print(type(features[2]))