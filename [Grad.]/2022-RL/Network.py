import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(state_size)

        # 3D input (atari games)
        if self.state_dim == 3:
            self.Conv1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.Conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

            self.fc1 = nn.Linear(self.calc_input_layer(), layer_size)
            self.fc2 = nn.Linear(layer_size, action_size)
        
        elif self.state_dim == 1:
            self.head = nn.Linear(self.input_shape[0], layer_size)
            self.fc1 = nn.Linear(layer_size, layer_size)
            self.fc2 = nn.Linear(layer_size, action_size)
    
    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)

        return x.flatten().shape[0]
    
    def forward(self, input):
        if self.state_dim == 3:
            x = F.relu(self.Conv1(input))
            x = F.relu(self.Conv2(x))
            x = F.relu(self.Conv3(x))
            x = x.view(input.size(0), -1)

        else:
            x = F.relu(self.head(x))
        
        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        return output