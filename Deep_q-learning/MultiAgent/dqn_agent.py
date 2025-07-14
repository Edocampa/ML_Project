import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Definition of Replay Buffer

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity) # deque discard the oldest element when full

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))


# sample a mini-batch of transitions
    def sample(self, batch_size=64):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# Definition of the network

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
    
# Definition of the agent with all hyperparameters

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
                 eps_decay_steps=250_000,
                 target_update_freq=150,
                 device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma     = gamma
        self.batch_size = batch_size

        # Epsilon-greedy schedule
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay_steps = eps_decay_steps
        #decrement applied at each step
        self.eps_delta = (eps_start - eps_end) / eps_decay_steps

        self.target_update = target_update_freq
        self.step_count    = 0

        # Networks
        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Optimizer  and MSE Loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(buffer_size)
        self.loss_fn   = nn.MSELoss()

    # Selection of the action

    def select_action(self, state):
        # Update epsilon until eps_min
        if self.step_count < self.eps_decay_steps:
            self.eps = max(self.eps_min, self.eps - self.eps_delta)

        # Exploration - Exploitation
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        state_v = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.online_net(state_v)
        return int(qvals.argmax(dim=1).item())
    
    # Learning step

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return None
        
        # Sample random batch from replay buffer
        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors to use Pytorch
        states      = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        actions     = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        dones       = torch.tensor(batch.done, device=self.device).unsqueeze(1).float()

        # Compute current Q and target Q
        q_pred = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Compute Loss and back-propagation

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def step(self, transition):

        self.replay.push(*transition)
        self.step_count += 1
        loss = self.optimize()

        # Periodic target network update (every target_update_steps)
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.online_net.state_dict())
