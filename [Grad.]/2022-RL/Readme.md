# DDQN with PyTorch

## Code structure
  - Agent.py : DQN Network를 2개 이용, **target network와 local network를 생성해 DDQN 구성**
  - Network.py : **기본적인 DQN Network 구성**
  - ReplayBuffer.py : **일정 분기마다 local network의 (s, a, r, s', a')를 기억**, 추후 target과 local에서 선택해 업데이트에 활용
  - Train.py : Atari game 중 **PongNoFrameskip-v4** 를 이용해 DDQN agent를 train
  - Wrapper.py : Agent가 행동할 env의 wrapper
    - gym에서 기본적으로 env를 구성할 수 있는 Wrapper를 제공하고, 이를 상속받아 env를 재 구성

<br>

## Hyper-Parameter
  - Optimizer : Adam with lr = 2e-4
  - Buffer size : 100,000
  - Target network update interval : every 1,000 frame

<br>

## Flow
  1. PongNoFrameskip-v4 환경 구성
  2. 구성한 환경의 action space, observable space를 이용해 DDQN Agent를 생성
     1. DDQN Agent는 2개의 DQN Network (target, local)로 구성되어 있음
     2. Epsilon-greedy 하게 policy improvement를 수행
     3. 지정한 분기마다 Replay Buffer로 (s, a, r, s', a')를 저장
     4. 처음으로 Buffer에 데이터가 저장 된 후, 이후로 Buffer의 값과 현재 값을 비교해 sample 유무를 선택
     5. 선택 후 TD-Error를 계산
  3. 모든 episode가 종료 된 후, 모델의 state를 './Trained' 경로 안에 저장

<br>

## Chart & Figures
![episode155_game](./episode155_capture.png)

    전체 약 300번의 episode 중 155번째 episode의 화면

![episode155_metric](./episode155_states.jpg)

    마찬가지로 155번째 episode에서의 reward, loss 그래프


<br>

## Some Notations & Reports
- **개인 노트북 환경(Windows)** 에서 작업하였으며, **GPU가 없는(no CUDA) 환경**이라 실험이 아직 진행중입니다.  <br>
  (220626 20:45 Ongoing)
- 학습 현황 chart는 **[wandb link](https://wandb.ai/happysky12/RL-DQN)** 에서 확인하실 수 있습니다.  <br>
  (Public으로 설정해두었으므로, 확인이 가능합니다.)
- 다음 링크들의 도움을 많이 받았으며, 참고 및 분석하며 새롭게 작성하였습니다.
  - [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
  - [XinJingHao](https://github.com/XinJingHao/DQN-DDQN-Pytorch)
  - [ShreeshaN](https://github.com/ShreeshaN/ReinforcementLearningTutorials/tree/master/DQN)
  - [BY571](https://github.com/BY571/DQN-Atari-Agents)
  - [p-christ](https://github.com/p-christ/deep-reinforcement-learning-algorithms-with-pytorch)
- 개인적으로 정말 귀중하고 여러 유익한 경험과 노하우를 알아갈 수 있었던 class(also project) 였습니다.