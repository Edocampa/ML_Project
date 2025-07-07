# dqn_agent.py
import random
from collections import deque, namedtuple
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Replay buffer ───────────────────────────────────────────────
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ── Q-network ───────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ── DQN agent (independent learner) ─────────────────────────────
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay: float = 1e-6,
        target_update_freq: int = 200,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update_freq
        self.step_count = 0

        # online / target network *per agente*
        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    # ── action selection ────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        state_v = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.online_net(state_v)
        return int(qvals.argmax(dim=1).item())

    # ── learning step ───────────────────────────────────────────
    def learn(
        self,
        transition: Tuple[np.ndarray, int, float, np.ndarray, bool]
    ):
        self.replay.push(*transition)
        self.step_count += 1
        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states_np      = np.stack([t.state      for t in batch])
        next_states_np = np.stack([t.next_state for t in batch])

        states      = torch.from_numpy(states_np).float().to(self.device)
        actions     = torch.tensor([t.action  for t in batch],
                                   device=self.device).unsqueeze(1)
        rewards     = torch.tensor([t.reward for t in batch],
                                   device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)
        dones       = torch.tensor([t.done   for t in batch],
                                   device=self.device).unsqueeze(1).float()

        q_pred = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_tar  = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_pred, q_tar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
