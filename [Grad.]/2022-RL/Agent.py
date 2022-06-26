import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from Network import DQN
from ReplayBuffer import Buffer

class DDQNAgent: 
    def __init__(self, in_channels = 1, action_space = [], USE_CUDA = False, memory_size = 10000, epsilon  = 1, lr = 1e-4):
        """
        DQN network를 기반으로 target, local 총 2개의 Q-network 구성
        """
        self.epsilon = epsilon 
        self.action_space = action_space
        self.memory_buffer = Buffer(memory_size)
        self.DQN = DQN(in_channels = in_channels, num_actions = action_space.n)
        self.DQN_target = DQN(in_channels = in_channels, num_actions = action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        self.optimizer = optim.Adam(self.DQN.parameters(), lr=lr, eps=0.001)

    def observe(self, lazyframe):
        # Frame으로 부터 관측된 현재 state
        state = torch.from_numpy(lazyframe._force().transpose(2,0,1)[None]/255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        # 현재 state의 Q-value 계산
        q_values = self.DQN(state)
        return q_values
    
    def act(self, state, epsilon = None):
        """
        Epsilon-greedy policy에 따라 action을 sample.
        random 또는 가장 높은 Q(s,a)를 선택.
        """
        if epsilon is None: epsilon = self.epsilon

        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton
    
    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ 
        TD-Loss 계산
        """
        actions = torch.tensor(actions).long()    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).bool()  # shape: [batch_size]
        
        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # 현재 state에서 모든 action에 대한 Q-value 계산
        predicted_qvalues = self.DQN(states)

        # 선택한 action에서의 Q-value 선택
        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ]

        # 다음 state의 모든 action들에 대한 Q-value를 계산
        predicted_next_qvalues_current = self.DQN(next_states)
        predicted_next_qvalues_target = self.DQN_target(next_states)

        # 앞서 구한 Q-value를 이용해 V*를 계산
        next_state_values =  predicted_next_qvalues_target.gather(1, torch.max(predicted_next_qvalues_current, 1)[1].unsqueeze(1)).squeeze(1)
        
        # Loss 계산을 위한 target Q-value 계산. 
        target_qvalues_for_actions = rewards + gamma * next_state_values

        # 마지막 state는 s'가 없는 형태 : Q(s,a) = r(s,a)
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # TD-Loss 계산
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        return loss
    
    def sample_from_buffer(self, batch_size):
        """
        Replay Buffer로부터 (s, a, r, s', a')를 random sampling
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done= data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size):
        """
        실제 학습이 이뤄지는 부분
        """
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return(td_loss.item())
        else:
            return(0)
    