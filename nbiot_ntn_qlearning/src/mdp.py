"""
mdp.py — observable state, action space and reward for the RACH controller.

Everything here uses only what a gNB can observe per NPRACH pool k (R_k subcarriers):
    idle-looking subcarriers, successes (Msg4 sent), collided subcarriers flagged by an early
    collision detector (probability P_cd, [18],[19]), and Msg3 failures after an RAR.
From these a slotted-ALOHA estimator recovers the per-subcarrier attempt rate lambda:
    collided-subcarrier fraction  f_c = (flagged / P_cd) / R = 1 - e^-l (1 + l)
    detected singletons           s   = succ + Msg3 failures not explained by collisions
                                  l  >= solution of l e^-l = s / R   (lower branch)
and from lambda the estimated misses  m = max(0, R l e^-l - s),
the estimated colliding meters        c = R l (1 - e^-l),
and the estimated timeouts            t = max(0, Msg3 failures - unflagged detected collisions).

State  s = (load bin, elevation bin, pass-phase bin, dominant-failure-cause bin)
Action a = (barring factor p, backoff indicator BI, repetition profile)
Reward r = [w_s n_s - w_c c - w_m m - w_t t] / R_active - w_e * airtime_share
   (lumped ablation: the three failure weights are replaced by their mean)
"""

import numpy as np

_LGRID = np.linspace(0.0, 12.0, 6001)
_FC = 1.0 - np.exp(-_LGRID) * (1.0 + _LGRID)          # increasing in lambda
_LO = _LGRID[_LGRID <= 1.0]
_FS = _LO * np.exp(-_LO)                                # increasing on [0, 1]


def _lambda_from_collisions(fc):
    return float(np.interp(min(max(fc, 0.0), _FC[-1]), _FC, _LGRID))


def _lambda_from_singles(fs):
    return float(np.interp(min(max(fs, 0.0), _FS[-1]), _FS, _LO))


_ML_L = np.linspace(0.0, 6.0, 121)[:, None]
_ML_P = np.linspace(0.05, 1.0, 20)[None, :]
_E = np.exp(-_ML_L)
_P_SINGLE = _ML_L * _E * _ML_P
_P_COLL = np.broadcast_to(1.0 - _E * (1.0 + _ML_L), _P_SINGLE.shape)
_P_IDLE = 1.0 - _P_SINGLE - _P_COLL
_LOG = [np.log(np.maximum(x, 1e-12)) for x in (_P_IDLE, _P_SINGLE, _P_COLL)]


def _lambda_ml(R, s_det, c_sc):
    """Grid maximum-likelihood lambda from (idle, detected-single, collided) subcarrier counts."""
    s_det = min(max(s_det, 0.0), R)
    c_sc = min(max(c_sc, 0.0), R - s_det)
    if c_sc <= 1e-9 and s_det <= 1e-9:
        return 0.0
    idle = R - s_det - c_sc
    ll = idle * _LOG[0] + s_det * _LOG[1] + c_sc * _LOG[2]
    i, _ = np.unravel_index(np.argmax(ll), ll.shape)
    return float(_ML_L[i, 0])


def estimate_pool(R, succ, flagged, msg3_fail, p_cd):
    """Estimate (attempts, colliding meters, misses, timeouts, lambda) for one pool."""
    if R <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    c_sc = flagged / p_cd
    unflagged = c_sc * (1.0 - p_cd)
    t_hat = max(0.0, msg3_fail - unflagged)
    s_det = succ + t_hat
    lam = _lambda_ml(R, s_det, c_sc)
    n_hat = R * lam
    m_hat = max(0.0, R * lam * np.exp(-lam) - s_det)
    c_hat = R * lam * (1.0 - np.exp(-lam))
    return n_hat, c_hat, m_hat, t_hat, lam


class RACH_MDP:
    def __init__(self, cfg, use_cause_state=True, use_satellite_state=True, lumped_reward=False):
        self.cfg = cfg
        self.use_cause_state = use_cause_state
        self.use_satellite_state = use_satellite_state
        self.lumped = lumped_reward
        self.actions = [(p, bi, j) for p in cfg["acb_levels"] for bi in cfg["bi_levels_ms"]
                        for j in range(len(cfg["rep_profiles"]))]
        self.n_actions = len(self.actions)
        self.nl = len(cfg["load_thresholds"]) + 1
        self.ne = len(cfg["elev_thresholds"]) + 1
        self.nph = len(cfg["phase_thresholds"]) + 1
        self.nc = 4
        self.n_states = self.nl * self.ne * self.nph * self.nc
        w = cfg
        self.w = (w["w_s"], w["w_c"], w["w_m"], w["w_t"], w["w_e"])
        if lumped_reward:
            wf = (w["w_c"] + w["w_m"] + w["w_t"]) / 3.0
            self.w = (w["w_s"], wf, wf, wf, w["w_e"])
        self.p_cd = float(cfg["ecd_detect_prob"])
        self.beta = float(cfg["estimator_ewma"])
        self.rep_max = max(max(p) for p in cfg["rep_profiles"])
        self.reset()

    # ------------------------------------------------------------------ estimator memory
    def reset(self):
        n_pools = len(self.cfg["ce_subcarriers"])
        self.ewma = np.zeros((n_pools, 3))     # succ, flagged, msg3_fail
        self.seen = np.zeros(n_pools, bool)
        self._est = np.zeros(5)

    def observe(self, rao):
        """Update smoothed per-pool observations; return the epoch reward (None if no pool active)."""
        R_act, num, air = 0.0, 0.0, 0.0
        ws, wc, wm, wt, we = self.w
        for pool in rao["pools"]:
            if not pool["active"]:
                continue
            k, R = pool["k"], pool["R"]
            obs = np.array([pool["succ"], pool["flagged"], pool["msg3_fail"]], float)
            self.ewma[k] = obs if not self.seen[k] else self.beta * obs + (1 - self.beta) * self.ewma[k]
            self.seen[k] = True
            n_hat, c_hat, m_hat, t_hat, _ = estimate_pool(R, *obs, self.p_cd)
            R_act += R
            num += ws * obs[0] - wc * c_hat - wm * m_hat - wt * t_hat
            air += n_hat * pool["reps"]
        if R_act == 0:
            return None
        self._est = None
        return num / R_act - we * air / (R_act * self.rep_max)

    def estimates(self):
        """Smoothed totals over pools: attempts, colliding meters, misses, timeouts, successes."""
        if self._est is not None:
            return self._est
        tot = np.zeros(5)
        for k, R in enumerate(self.cfg["ce_subcarriers"]):
            if self.seen[k]:
                n_hat, c_hat, m_hat, t_hat, _ = estimate_pool(R, *self.ewma[k], self.p_cd)
                tot += (n_hat, c_hat, m_hat, t_hat, self.ewma[k][0])
        self._est = tot
        return tot

    # ------------------------------------------------------------------ state / action
    def encode_state(self, elevation_deg, pass_progress):
        cfg = self.cfg
        n_hat, c_hat, m_hat, t_hat, succ = self.estimates()
        load = n_hat / float(sum(cfg["ce_subcarriers"]))
        lb = int(np.digitize(load, cfg["load_thresholds"]))
        if self.use_satellite_state:
            eb = int(np.digitize(elevation_deg, cfg["elev_thresholds"]))
            pb = int(np.digitize(pass_progress, cfg["phase_thresholds"]))
        else:
            eb = pb = 0
        cb = 0
        if self.use_cause_state:
            fails = c_hat + m_hat + t_hat
            total = fails + succ
            if total >= 1.0 and fails / total >= cfg["cause_fail_threshold"]:
                cb = 1 + int(np.argmax([c_hat, m_hat, t_hat]))
        return ((lb * self.ne + eb) * self.nph + pb) * self.nc + cb

    def decode_action(self, a):
        p, bi, j = self.actions[a]
        return p, bi, self.cfg["rep_profiles"][j]

    def encode_action(self, p, bi, profile_idx):
        return self.actions.index((p, bi, profile_idx))
