import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Replay transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        # Return a list of Transition tuples
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
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
                 buffer_size=100_000, 
                 batch_size=64,
                 gamma=0.95, 
                 lr=1e-3,
                 eps_start=1.0,
                 eps_end=0.01, 
                 eps_decay_steps= 500_000,
                 target_update_freq=1000, 
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        # epsilon schedule
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_delta = (eps_start - eps_end) / eps_decay_steps

        self.target_update = target_update_freq
        self.step_count = 0

        # networks
        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # decay epsilon
        if self.step_count < self.eps_decay_steps:
            self.eps = max(self.eps_min, self.eps - self.eps_delta)
        # epsilon-greedy
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        state_v = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.online_net(state_v)
        return int(qvals.argmax(dim=1).item())

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return None
        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        states = torch.FloatTensor(np.stack(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # current Q
        q_pred = self.online_net(states).gather(1, actions)
        # target Q
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def step(self, transition):
        self.replay.push(*transition)
        self.step_count += 1
        loss = self.optimize()
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.online_net.state_dict())