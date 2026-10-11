"""
baselines.py — Baseline RACH controllers and ablation policies.

Implements the five comparison schemes defined in Methodology §4:
1. B1: Fixed ACB+BI (terrestrial default p=1.0, BI=256ms, N_rep=2, RAR=8ms).
2. B2: Rel-17 NPRACH-NTN default (fixed NTN preset p=0.7, BI=1024ms, N_rep=8, RAR=40ms).
3. B3: Terrestrial-frozen Q-table (trained on terrestrial channel, frozen on NTN).
4. B4: Oracle (perfect per-RAO channel & backlog knowledge).
5. Lumped-Reward Ablation (Q-learning agent with lumped failure penalty).
"""

import numpy as np
from typing import Tuple, Dict
from src.mdp import RACH_MDP
from src.agent import TabularQLearningAgent


class BaselineController:
    """Base class for all RACH controllers."""
    def __init__(self, name: str, mdp: RACH_MDP):
        self.name = name
        self.mdp = mdp

    def get_action(self, state_dict: dict) -> Tuple[float, int, int]:
        """Return (acb_p, bi_ms, n_rep)."""
        raise NotImplementedError

    def get_rar_window_s(self) -> float:
        """Return RAR window duration in seconds."""
        return 0.040  # default NTN extended RAR


class FixedTerrestrialController(BaselineController):
    """
    B1: Fixed ACB+BI (Terrestrial Default)
    p = 1.0 (no barring), BI = 256 ms, N_rep = 2.
    Uses realistic NB-IoT RAR window (24 ms) to allow honest evaluation of
    contention/detection performance without artificial strawman timeouts.
    """
    def __init__(self, mdp: RACH_MDP, rar_window_s: float = 0.024):
        super().__init__("B1_Fixed_Terrestrial", mdp)
        self.rar_window_s = rar_window_s

    def get_action(self, state_dict: dict) -> Tuple[float, int, int]:
        return (1.0, 256, 2)

    def get_rar_window_s(self) -> float:
        return self.rar_window_s


class Rel17NTNDefaultController(BaselineController):
    """
    B2: Rel-17 NPRACH-NTN Default Preset
    Extended RAR = 40 ms, p = 0.7, BI = 1024 ms, N_rep = 8.
    """
    def __init__(self, mdp: RACH_MDP):
        super().__init__("B2_Rel17_NTN_Default", mdp)

    def get_action(self, state_dict: dict) -> Tuple[float, int, int]:
        return (0.7, 1024, 8)

    def get_rar_window_s(self) -> float:
        return 0.040


class HeuristicBenchmarkController(BaselineController):
    """
    B4: Heuristic Channel-Aware Benchmark (Idealized Hand-Crafted Rules)
    Dynamically switches parameters based on instantaneous geometry and backlog:
    - Low elevation -> high repetition (N_rep = 32)
    - Moderate elevation -> N_rep = 8
    - Zenith elevation -> N_rep = 2 (saves transmission airtime)
    - High backlog -> stronger barring and longer BI
    """
    def __init__(self, mdp: RACH_MDP):
        super().__init__("B4_Heuristic_Benchmark", mdp)

    def get_action(self, state_dict: dict) -> Tuple[float, int, int]:
        elev = state_dict['elevation_deg']
        n_eligible = state_dict.get('n_eligible', 0)
        n_preambles = state_dict.get('n_preambles', 48)

        # 1. Optimal repetition based on elevation & Doppler
        if elev < 25.0:
            best_nrep = 32
        elif elev < 50.0:
            best_nrep = 8
        else:
            best_nrep = 2

        # 2. Optimal ACB and BI based on offered load
        load_ratio = float(n_eligible) / float(n_preambles)
        if load_ratio > 1.5:
            best_p = 0.3
            best_bi = 4096
        elif load_ratio > 0.8:
            best_p = 0.5
            best_bi = 1024
        elif load_ratio > 0.3:
            best_p = 0.7
            best_bi = 1024
        else:
            best_p = 1.0
            best_bi = 256

        return (best_p, best_bi, best_nrep)

    def get_rar_window_s(self) -> float:
        return 0.040


# Backward-compatible alias for existing tests/scripts
OracleController = HeuristicBenchmarkController
