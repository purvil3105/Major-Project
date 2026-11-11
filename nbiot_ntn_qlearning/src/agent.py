"""
agent.py — Tabular Q-learning controller with cold-start and warm-start modes.

Implements standard Q-learning update:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
with epsilon-greedy exploration and exponential decay.

Supports:
- Cold-start: Q-table initialized to zeros.
- Warm-start: Q-table loaded from offline pre-trained table.
- Terrestrial pre-training routine to generate the warm-start table.
"""

import os
import numpy as np
from typing import Tuple, Optional
from src.mdp import RACH_MDP


class TabularQLearningAgent:
    """
    Lightweight Tabular Q-Learning Agent for NB-IoT NTN access control.
    """

    def __init__(
        self,
        mdp: RACH_MDP,
        alpha: float = 0.12,
        gamma: float = 0.90,
        eps0: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.995,
        seed: int = 200,
        is_warm_start: bool = False,
        warmstart_path: Optional[str] = None
    ):
        self.mdp = mdp
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps0)
        self.eps_min = float(eps_min)
        self.eps_decay = float(eps_decay)
        self.rng = np.random.default_rng(seed)

        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions

        # Initialize Q-table
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.is_warm_start = is_warm_start

        if is_warm_start:
            self.load_q_table(warmstart_path)
            # For warm-start, start with lower exploration rate (fine-tuning)
            self.eps = max(0.15, self.eps_min)

        # Statistics tracking
        self.update_count = 0
        self.total_reward = 0.0

    def select_action(self, state_idx: int, greedy_only: bool = False) -> int:
        """
        Choose an action index using epsilon-greedy policy.
        """
        if not greedy_only and (self.rng.random() < self.eps):
            return int(self.rng.integers(0, self.n_actions))
        else:
            # Greedy choice with random tie-breaking
            q_row = self.q_table[state_idx]
            max_val = np.max(q_row)
            best_actions = np.flatnonzero(np.isclose(q_row, max_val, atol=1e-8))
            return int(self.rng.choice(best_actions))

    def update(
        self,
        state_idx: int,
        action_idx: int,
        reward: float,
        next_state_idx: int,
        done: bool = False
    ) -> float:
        """
        Perform standard Q-learning Bellman update.
        
        Returns TD error.
        """
        q_current = self.q_table[state_idx, action_idx]
        if done:
            target = reward
        else:
            q_next_max = np.max(self.q_table[next_state_idx])
            target = reward + self.gamma * q_next_max

        td_error = target - q_current
        self.q_table[state_idx, action_idx] += self.alpha * td_error

        # Decay exploration
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self.update_count += 1
        self.total_reward += reward

        return float(td_error)

    def save_q_table(self, file_path: str):
        """Save Q-table to .npy file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        np.save(file_path, self.q_table)

    def load_q_table(self, file_path: Optional[str] = None):
        """Load pre-trained Q-table."""
        if file_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(project_root, "results", "warmstart_qtable.npy")
        if os.path.exists(file_path):
            loaded = np.load(file_path)
            if loaded.shape == self.q_table.shape:
                self.q_table = loaded.copy()
            else:
                print(f"Warning: Q-table shape mismatch: {loaded.shape} vs {self.q_table.shape}")
        else:
            print(f"Warning: Warmstart file not found: {file_path}, falling back to zeros.")
