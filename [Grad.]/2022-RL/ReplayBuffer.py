import numpy as np
from collections import deque, namedtuple
import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1, parallel_env=4):
        """
        Initialize replay buffer

        Parameters
        ==========
            buffer_size (int) : buffer's maximum size
            batch_size (int) : size of training batch
            seed (int) : random seed
        """

        self.device = device
        self.memory = deque(maxlen=buffer_size) # Make buffer memory by deque
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done']) # For comfortable experience access
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for i in range(parallel_env)]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add new experience to buffer
        """
        if self.iter_ == self.parallel_env:
            self.iter_ = 0

        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done)) # Add experiences to buffer

        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[self.iter_])
            e = self.experience(state, action, reward, next_state, done) 
            self.memory.append(e)
        
        self.iter_ += 1
    
    def calc_multistep_return(self, n_step_buffer):
        """
        Calculate n-step return
        """
        return_ = 0

        for i in range(self.n_step):
            return_ += (self.gamma**i) * n_step_buffer[i][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], return_, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)