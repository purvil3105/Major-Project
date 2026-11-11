"""
agent.py — tabular Q-learning with epsilon-greedy exploration.

    Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]

Cold start: Q = 0. Warm start: Q loaded from a table pre-trained on *other* passes.
Memory: |S| x |A| float64 = 108 x 60 x 8 B = 51.8 KB.
"""

import os

import numpy as np


class TabularQLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, eps0, eps_min, eps_decay, rng,
                 q_table=None):
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.eps, self.eps_min, self.eps_decay = float(eps0), float(eps_min), float(eps_decay)
        self.rng = rng
        self.q = np.zeros((n_states, n_actions)) if q_table is None else np.array(q_table, float)
        self.updates = 0

    def select_action(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.q.shape[1]))
        row = self.q[s]
        best = np.flatnonzero(row >= row.max() - 1e-12)
        return int(self.rng.choice(best))

    def update(self, s, a, r, s_next, done=False):
        target = r if done else r + self.gamma * self.q[s_next].max()
        td = target - self.q[s, a]
        self.q[s, a] += self.alpha * td
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self.updates += 1
        return float(td)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, self.q)
