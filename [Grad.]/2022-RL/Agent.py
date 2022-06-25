from ReplayBuffer import ReplayBuffer
from Network import QNetwork

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Interact with env, learn from env
class Agent():
    def __init__(self, state_size, action_size, layer_size, n_step, BATCH_SIZE, BUFFER_SIZE, LR, TAU, GAMMA, device, seed, worker):
        """
        Initialize agent

        Parameters
        ==========
            state_size (int) : dim of each state
            action_size (int) : dim of each action
            layer_size (int) : size of hidden layer
            BATCH_SIZE (int) : size of training batch
            BUFFER_SIZE (int) : size of replay buffer
            LR (float) : learning rate
            TAU (float) : tau for updating network weights
            GAMMA (float) : discount factor
            device (str) : used device to compute (CPU, GPU)
            seed (int) : random seed
        """
        self.seed = random.seed(seed)
        self.torch_seed = torch.manual_seed(seed)
        self.device = device
        self.worker = worker

        self.state_size = state_size
        self.action_size = action_size
        self.eta = 0.1
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = 1 # update frequency
        self.BATCH_SIZE = BATCH_SIZE * worker
        self.Q_updates = 0
        self.n_step = n_step
        self.action_step = 4
        self.last_action = None

        # Local Q-Net & Target Q-Net
        self.Q_network_local = QNetwork(state_size, action_size, layer_size, n_step, seed).to(device)
        self.Q_network_target = QNetwork(state_size, action_size, layer_size, n_step, seed).to(device)

        # Replay Buffer
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step, parallel_env=self.worker)

        self.optimizer = optim.Adam(self.Q_network_local.parameters(), lr=LR)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done, writer):
        # Save experiences to buffer
        self.buffer.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in buffer, get random subset and learn
            experiences = self.buffer.sample()
            