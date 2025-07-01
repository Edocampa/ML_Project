import numpy as np
from collections import defaultdict

class BaseAgent:
    def select_action(self, state):
        raise NotImplementedError

    def learn(self, transition):
        raise NotImplementedError


class QLearningAgent(BaseAgent):
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.actions = actions  # e.g., [0, 1, 2, 3]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_table[state])

    def learn(self, transition):
        state, action, reward, next_state, done = transition

        max_future_q = 0 if done else np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
