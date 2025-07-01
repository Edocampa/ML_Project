import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Transizione
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Replay buffer FIFO
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# La Q-Network base
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

# L’agente DQN
class DQNAgent:
    def __init__(self, state_dim, n_actions,
                 buffer_size=100000, batch_size=64,
                 gamma=0.99, lr=1e-3,
                 eps_start=1.0, eps_end=0.05, eps_decay=1e-4,
                 target_update_freq=1000,
                 device=None):
        self.device = device or torch.device('cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update_freq
        self.step_count = 0

        # reti online e target
        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """ε-greedy: state è np.array float32"""
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.online_net(st)
        return int(qvals.argmax().item())

    def learn(self, transition):
        """transition = (s,a,r,s2,done)"""
        self.replay.push(*transition)
        self.step_count += 1

        # decay ε
        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        if len(self.replay) < self.batch_size:
            return

        # sample minibatch
        batch = self.replay.sample(self.batch_size)
        S  = torch.tensor([t.state      for t in batch], device=self.device)
        A  = torch.tensor([t.action     for t in batch], device=self.device).unsqueeze(1)
        R  = torch.tensor([t.reward     for t in batch], device=self.device).unsqueeze(1)
        S2 = torch.tensor([t.next_state for t in batch], device=self.device)
        D  = torch.tensor([t.done       for t in batch], device=self.device).unsqueeze(1).float()

        # predizione Q(s,a)
        Qpred = self.online_net(S).gather(1, A)

        # target: r + γ max_a Q_target(s',a)
        with torch.no_grad():
            Qnext = self.target_net(S2).max(1, keepdim=True)[0]
            Qtarget = R + self.gamma * Qnext * (1 - D)

        # loss e ottimizzazione
        loss = self.loss_fn(Qpred, Qtarget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # aggiorna target network ogni N passi
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
