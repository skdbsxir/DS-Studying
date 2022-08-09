"""
Train GCN model
"""
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from models import GCN
from utils import accuracy