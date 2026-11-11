"""
mdp.py — State encoding, action decoding, and cause-separated reward formulation.

State s = (load_bin, elev_bin, doppler_bin, pass_phase_bin)
Action a = (acb_level, bi_level, nrep_level)
Reward r = cause-separated reward (Methodology §3.3)
"""

import numpy as np
from typing import Tuple, Dict, List


class RACH_MDP:
    """
    MDP definition for NB-IoT NTN Random Access Optimization.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Action parameters
        self.acb_levels = list(cfg['acb_levels'])
        self.bi_levels = list(cfg['bi_levels_ms'])
        self.nrep_levels = list(cfg['nrep_levels'])

        self.n_acb = len(self.acb_levels)
        self.n_bi = len(self.bi_levels)
        self.n_nrep = len(self.nrep_levels)
        self.n_actions = self.n_acb * self.n_bi * self.n_nrep

        # Pre-build action lookup table: index -> (acb, bi, nrep)
        self.action_table = []
        for acb in self.acb_levels:
            for bi in self.bi_levels:
                for nrep in self.nrep_levels:
                    self.action_table.append((acb, bi, nrep))

        # State discretization parameters
        self.load_bins = int(cfg['load_bins'])
        self.elev_bins = int(cfg['elev_bins'])
        self.doppler_bins = int(cfg['doppler_bins'])
        self.pass_phase_bins = int(cfg['pass_phase_bins'])

        self.load_thresholds = list(cfg['load_thresholds'])
        self.elev_thresholds = list(cfg['elev_thresholds'])
        self.doppler_threshold = float(cfg['doppler_threshold'])
        self.pass_phase_thresholds = list(cfg['pass_phase_thresholds'])

        self.n_states = (self.load_bins * self.elev_bins *
                         self.doppler_bins * self.pass_phase_bins)

        # Reward weights
        self.w_s = float(cfg['w_s'])
        self.w_c = float(cfg['w_c'])
        self.w_m = float(cfg['w_m'])
        self.w_t = float(cfg['w_t'])
        self.w_d = float(cfg['w_d'])
        self.d_ref_ms = float(cfg['d_ref_ms'])

        # Lumped failure penalty for ablation
        self.w_lumped = (self.w_c + self.w_m + self.w_t) / 3.0

    def encode_state(
        self,
        load_ratio: float,
        elevation_deg: float,
        residual_doppler_hz: float,
        pass_progress: float
    ) -> int:
        """
        Discretize state variables into a single flat state index in [0, n_states - 1].
        
        Parameters
        ----------
        load_ratio : float
            Estimated channel load ratio (transmitting devices / preambles).
        elevation_deg : float
            Satellite elevation angle in degrees.
        residual_doppler_hz : float
            Residual Doppler offset in Hz.
        pass_progress : float
            Fraction of pass elapsed in [0.0, 1.0].
        """
        # 1. Load bin [0..load_bins-1]
        l_bin = int(np.digitize(load_ratio, self.load_thresholds))
        l_bin = min(l_bin, self.load_bins - 1)

        # 2. Elevation bin [0..elev_bins-1]
        e_bin = int(np.digitize(elevation_deg, self.elev_thresholds))
        e_bin = min(e_bin, self.elev_bins - 1)

        # 3. Doppler bin [0..1]
        d_bin = 1 if residual_doppler_hz > self.doppler_threshold else 0

        # 4. Pass phase bin [0..pass_phase_bins-1]
        p_bin = int(np.digitize(pass_progress, self.pass_phase_thresholds))
        p_bin = min(p_bin, self.pass_phase_bins - 1)

        # Linear index
        state_idx = (
            l_bin * (self.elev_bins * self.doppler_bins * self.pass_phase_bins) +
            e_bin * (self.doppler_bins * self.pass_phase_bins) +
            d_bin * (self.pass_phase_bins) +
            p_bin
        )
        return int(state_idx)

    def decode_action(self, action_idx: int) -> Tuple[float, int, int]:
        """Convert flat action index in [0, n_actions - 1] to (acb_p, bi_ms, n_rep)."""
        return self.action_table[action_idx]

    def encode_action(self, acb_p: float, bi_ms: int, n_rep: int) -> int:
        """Convert (acb_p, bi_ms, n_rep) to flat action index."""
        try:
            return self.action_table.index((acb_p, bi_ms, n_rep))
        except ValueError:
            # Nearest match
            i_acb = int(np.argmin([abs(x - acb_p) for x in self.acb_levels]))
            i_bi = int(np.argmin([abs(x - bi_ms) for x in self.bi_levels]))
            i_rep = int(np.argmin([abs(x - n_rep) for x in self.nrep_levels]))
            return i_acb * (self.n_bi * self.n_nrep) + i_bi * self.n_nrep + i_rep

    def compute_reward(self, rao_stats: dict, use_lumped_ablation: bool = False) -> float:
        """
        Compute cause-separated reward r (Methodology §3.3).
        
        r = (w_s*n_s - w_c*n_c - w_m*n_m - w_t*n_t) / n - w_d * (d_mean / d_ref)
        """
        n_trans = rao_stats['n_trans']
        if n_trans == 0:
            return 0.0

        n_s = float(rao_stats.get('n_success', 0))
        n_c = float(rao_stats.get('n_collision', 0))
        n_t = float(rao_stats.get('n_timeout', 0))

        # Miss count: either observable estimator or ground truth benchmark
        use_estimated = self.cfg.get('use_estimated_misses', True)
        if use_estimated and 'p_d' in rao_stats:
            p_d = max(0.01, min(0.999, rao_stats['p_d']))
            # Observable estimator: n_m_est = n_s * (1 - P_d) / P_d
            n_m = float(n_s) * (1.0 - p_d) / p_d
        else:
            n_m = float(rao_stats.get('n_miss', 0))

        # Average delay term for successfully served devices in this RAO
        recent_delays = rao_stats.get('recent_delays', [])
        if len(recent_delays) > 0:
            d_mean_ms = float(np.mean(recent_delays))
            delay_penalty = self.w_d * min(d_mean_ms / self.d_ref_ms, 3.0)
        else:
            delay_penalty = 0.0

        if use_lumped_ablation:
            # Lumped failure penalty: treats miss, collision, timeout identically
            n_fail = n_c + n_m + n_t
            r = (self.w_s * n_s - self.w_lumped * n_fail) / float(n_trans) - delay_penalty
        else:
            # Cause-separated reward (THE PROPOSED NOVELTY)
            r = (self.w_s * n_s - self.w_c * n_c - self.w_m * n_m - self.w_t * n_t) / float(n_trans) - delay_penalty

        return float(r)
