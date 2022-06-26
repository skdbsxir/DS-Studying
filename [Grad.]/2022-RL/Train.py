import math
import numpy as np
import torch
import wandb

from Wrapper import make_atari, wrap_deepmind
from Agent import DDQNAgent

# wandb setting
wandb.init(project="RL-DQN", entity="happysky12")

# Training DQN in PongNoFrameskip-v4 
env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, scale = False, frame_stack=True)

# Hyper Parameters
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
frames = 500000
USE_CUDA = False
learning_rate = 2e-4
max_buff = 100000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000 # 1000 frame마다 print
log_interval = 1000
learning_start = 10000
win_reward = 18     # Pong-v4
win_break = True

wandb.config = {
    'gamma' : 0.99,
    'epsilon_max' : 1,
    'epsilon_min' : 0.01,
    'frames' : 50000,
    'lr' : 2e-4,
    'buffer' : 100000,
}

# Agent 초기 설정 -> action space, state channel 전달
action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
state_channel = env.observation_space.shape[2]
agent = DDQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate)
# wandb.watch(agent) 

frame = env.reset()

episode_reward = 0
all_rewards = []
losses = []
episode_num = 0
is_win = False

# epsilon-greedy 감쇠
epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(-1. * frame_idx / eps_decay)

# Train start
for i in range(frames):
    epsilon = epsilon_by_frame(i)
    state_tensor = agent.observe(frame) # state 관측
    action = agent.act(state_tensor, epsilon) # epsilon policy에 따라 action 선택
    
    next_frame, reward, done, _ = env.step(action) # 선택한 action에 따른 보상, state get
    
    episode_reward += reward
    agent.memory_buffer.push(frame, action, reward, next_frame, done) # 해당 내용을 buffer에 push
    frame = next_frame
    
    loss = 0
    if agent.memory_buffer.size() >= learning_start:
        # 학습 수행
        loss = agent.learn_from_experience(batch_size)
        losses.append(loss)

    if i % print_interval == 0:
        # 진행중인 reward현황, loss현황, episode 수 출력
        print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
        
    if i % update_tar_interval == 0:
        agent.DQN_target.load_state_dict(agent.DQN.state_dict())
    
    if done:
        # 게임이 종료된 경우(승리 or 패배), env를 reset
        frame = env.reset()
        all_rewards.append(episode_reward) # 해당 episode에서의 reward를 가지고
        episode_reward = 0
        episode_num += 1
        avg_reward = float(np.mean(all_rewards[-100:])) # episode에서의 평균 reward를 계산

        # for wandb graph plotting
        wandb.log({
            # 'step' : episode_num,
            'epsilon' : epsilon,
            'loss' : loss,
            # 'reward' : np.mean(all_rewards[-10:])
            'reward' : avg_reward
        })

# 종료된 agent의 state 저장
torch.save(agent.DQN.state_dict(), f'./Trained/DDQN_dict_{frames}.pth')