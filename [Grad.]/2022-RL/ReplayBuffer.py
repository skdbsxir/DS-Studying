import numpy as np
import random

class Buffer(object):
    def __init__(self, memory_size=1000):
        """
        Initialize Buffer
        """
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0
        
    def push(self, state, action, reward, next_state, done):
        """
        Buffer에 데이터 push
        """
        data = (state, action, reward, next_state, done)

        # Buffer가 아직 가득차있지 않은 경우
        if len(self.buffer) <= self.memory_size:
            self.buffer.append(data)

        # Buffer가 가득 찬 경우 -> replace
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        """
        Buffer로부터 random하게 sample
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones
    
    def size(self):
        return len(self.buffer)