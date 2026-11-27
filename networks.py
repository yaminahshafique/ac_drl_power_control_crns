
"""Actor and Critic Networks for DDPG and TD-3
 
Architecture from experimental details section of the DDPG 
paper "Continuous control with deep reinforcement learning"
Lillicrap et. al. 2015


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, action_max):
    super(Actor, self).__init__()
    self.action_max = action_max
    self.device = device

    self.linear1 = nn.Linear(state_dim, 64)
    self.linear2 = nn.Linear(64, 64)
    self.linear3 = nn.Linear(64, action_dim)

  def forward(self, state):
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))
    x = torch.tanh(self.linear3(x))

    action_max = torch.as_tensor(
        self.action_max,
        dtype=x.dtype,
        device=x.device,
    )

    return x * action_max

class Critic(nn.Module):
  def __init__(self, state_dim, action_dim, trainable = True):
    super(Critic, self).__init__()
    self.device = device
    self.W_s = nn.Parameter(torch.randn(state_dim, 64), requires_grad = trainable)
    self.W_a = nn.Parameter(torch.randn(action_dim, 64), requires_grad = trainable)
    self.b1 = nn.Parameter(torch.randn(1, 64), requires_grad = trainable)
    self.linear1 = nn.Linear(64, 64)
    self.linear2 = nn.Linear(64, 1)

  def forward(self, state, action):
    x = F.relu(torch.matmul(state, self.W_s) + torch.matmul(action, self.W_a) + self.b1)
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    return x
