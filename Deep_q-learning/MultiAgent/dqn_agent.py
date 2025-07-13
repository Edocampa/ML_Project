# dqn_agent.py (multi-agent friendly, same style as single-agent)
import random
from collections import deque, namedtuple
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ───────────────────────────────── Replay buffer ─────────────────────────────
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int = 64) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ────────────────────────────────── Q-network ────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ──────────────────────────────── DQN Agent ──────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.95,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 250_000,
        target_update_freq: int = 150,       # un po' più stretto per scenari non stazionari
        device: torch.device | None = None,
    ):
        # Device
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Hyper-params
        self.n_actions      = n_actions
        self.gamma          = gamma
        self.batch_size     = batch_size
        self.target_update  = target_update_freq

        # ε-greedy schedule
        self.eps            = eps_start
        self.eps_min        = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_delta      = (eps_start - eps_end) / eps_decay_steps

        # Networks
        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Opt & buffer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(buffer_size)
        self.loss_fn   = nn.MSELoss()

        # Counters
        self.step_count = 0

    # ───────────────────────── action selection ──────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        # Aggiorna ε (una sola volta per step ambiente)
        if self.step_count < self.eps_decay_steps:
            self.eps = max(self.eps_min, self.eps - self.eps_delta)

        if random.random() < self.eps:
            return random.randrange(self.n_actions)

        state_v = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.online_net(state_v)
        return int(qvals.argmax(dim=1).item())

    # ──────────────────────────── training API ───────────────────────────────
    def optimize(self):
        if len(self.replay) < self.batch_size:
            return None

        # Sample batch
        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Tensors
        states      = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        actions     = torch.tensor(batch.action,  device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward,  device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(self.device)
        dones       = torch.tensor(batch.done,    device=self.device).unsqueeze(1).float()

        # Q-learning targets
        q_pred = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            q_next   = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Back-prop
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

    # ─────────────── helper di salvataggio/restore (facoltativi) ──────────────
    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())
