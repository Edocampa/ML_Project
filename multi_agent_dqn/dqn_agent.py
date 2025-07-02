import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 state_dim,
                 n_actions,
                 buffer_size=100000,
                 batch_size=64,
                 gamma=0.99,
                 lr=1e-3,
                 eps_start=1.0,
                 eps_end=0.05,
                 eps_decay=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNetwork(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randrange(self.net.net[-1].out_features)
        else:
            state_v = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.net(state_v)
            return int(q_vals.argmax(dim=1).item())

    def learn(self, transition):
        # Unpack and store
        self.buffer.push(*transition)
        # Decay epsilon
        self.eps = max(self.eps_end, self.eps - self.eps_decay)

        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor([t.state for t in batch]).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_pred = self.net(states).gather(1, actions)
        # target: r + γ max_a' Q(s', a')
        with torch.no_grad():
            q_next = self.net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()