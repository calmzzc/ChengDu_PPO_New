import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

LR_v = 3e-4
LR_pi = 3e-4


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, 1)
        self.sigma = nn.Linear(hidden_dim, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)

    def forward(self, state):
        x = self.actor(state)
        mu = torch.tanh(self.mu(x)) * 2
        sigma = F.softplus(self.sigma(x)) + 0.001
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=LR_v)

    def forward(self, state):
        value = self.critic(state)
        return value
