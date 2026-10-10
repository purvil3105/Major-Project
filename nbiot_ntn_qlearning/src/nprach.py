"""
nprach.py — Monte-Carlo NB-IoT NPRACH contention engine with failure taxonomy.

Implements:
1. 3GPP TR 37.868 Beta(3,4) synchronized burst traffic over 10 s window.
2. Step-by-step RAO contention loop over the LEO satellite pass.
3. ACB barring check (TS 36.331) against barring factor p.
4. Contention resolution over R = 48 preambles (TS 36.211).
5. 4-way mutually exclusive failure taxonomy (Methodology §2.3):
   - SUCCESS: allowed, unique preamble, detected, in-window.
   - COLLISION: >= 2 devices chose the same preamble.
   - DETECTION MISS: unique preamble cleanly sent but missed (1 - P_d).
   - TIMEOUT: unique and detected, but 2*tau + t_proc > T_RAR.
6. Backoff Indicator (BI) retry scheduling (TS 36.321) up to preambleTransMax.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.channel import elevation_to_snr_db, compute_detection_probability


class DeviceState:
    """Internal state tracking for an NB-IoT meter."""
    __slots__ = ('dev_id', 'arrival_time_s', 'next_attempt_s', 'trans_count',
                 'is_active', 'is_served', 'is_dropped', 'access_delay_s',
                 'last_failure_cause')

    def __init__(self, dev_id: int, arrival_time_s: float):
        self.dev_id = dev_id
        self.arrival_time_s = arrival_time_s
        self.next_attempt_s = arrival_time_s
        self.trans_count = 0
        self.is_active = False
        self.is_served = False
        self.is_dropped = False
        self.access_delay_s = np.nan
        self.last_failure_cause = None


class NPRACHSimulator:
    """
    NPRACH Event-driven RAO contention simulator.
    """

    def __init__(self, cfg: dict, trajectory: dict, seed: int = 200):
        self.cfg = cfg
        self.traj = trajectory
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # Protocol constants from config
        self.n_preambles = int(cfg['n_preambles'])
        self.preamble_trans_max = int(cfg['preamble_trans_max'])
        self.processing_time_s = float(cfg['processing_time_ms']) / 1000.0
        self.scs_hz = float(cfg['scs_hz'])

        # Trajectory arrays (1-second sampled)
        self.traj_time = trajectory['time_s']
        self.traj_elev = trajectory['elevation_deg']
        self.traj_delay_s = trajectory['delay_s']
        self.traj_doppler = trajectory['raw_doppler_hz']
        self.pass_duration_s = float(trajectory['duration_s'])

        # Pass interpolators
        self._setup_orbit_lookup()

    def _setup_orbit_lookup(self):
        """Pre-compute quick lookups for pass trajectory."""
        self._t_max = self.traj_time[-1]

    def get_orbit_at_time(self, t_s: float) -> Tuple[float, float, float]:
        """Return (elevation_deg, delay_s, raw_doppler_hz) at time t_s."""
        t_clamped = np.clip(t_s, self.traj_time[0], self._t_max)
        elev = np.interp(t_clamped, self.traj_time, self.traj_elev)
        delay = np.interp(t_clamped, self.traj_time, self.traj_delay_s)
        doppler = np.interp(t_clamped, self.traj_time, self.traj_doppler)
        return float(elev), float(delay), float(doppler)

    def generate_traffic(self, n_devices: int, traffic_model: str = "beta") -> List[DeviceState]:
        """
        Generate activation arrival times for n_devices.
        
        Beta(3,4) over activation_window_s (default 10s) as per 3GPP TR 37.868.
        """
        win_s = float(self.cfg.get('activation_window_s', 10.0))
        if traffic_model == "beta":
            a = float(self.cfg.get('beta_a', 3))
            b = float(self.cfg.get('beta_b', 4))
            arrivals = self.rng.beta(a, b, size=n_devices) * win_s
        elif traffic_model == "poisson":
            arrivals = self.rng.uniform(0.0, win_s, size=n_devices)
        else:
            arrivals = self.rng.uniform(0.0, win_s, size=n_devices)

        arrivals.sort()
        devices = [DeviceState(dev_id=i, arrival_time_s=arrivals[i]) for i in range(n_devices)]
        return devices

    def simulate_rao(
        self,
        t_now_s: float,
        devices: List[DeviceState],
        acb_p: float,
        bi_ms: float,
        n_rep: int,
        residual_doppler_hz: float,
        rar_window_s: float
    ) -> Dict:
        """
        Execute one RAO at time t_now_s.
        
        Returns RAO outcome metrics dictionary.
        """
        elev, delay_s, _ = self.get_orbit_at_time(t_now_s)
        snr_db = elevation_to_snr_db(
            elev,
            snr_edge_db=self.cfg['snr_edge_db'],
            snr_zenith_db=self.cfg['snr_zenith_db'],
            elev_min_deg=self.cfg['elev_mask_deg']
        )
        p_d = compute_detection_probability(snr_db, residual_doppler_hz, n_rep, self.scs_hz)

        # 1. Identify backlogged eligible devices:
        # device is not served, not dropped, and arrival <= t_now_s and next_attempt_s <= t_now_s
        eligible = []
        for dev in devices:
            if not dev.is_served and not dev.is_dropped:
                if dev.arrival_time_s <= t_now_s and dev.next_attempt_s <= t_now_s + 1e-4:
                    eligible.append(dev)

        n_eligible = len(eligible)
        if n_eligible == 0:
            return {
                't_s': t_now_s,
                'elevation_deg': elev,
                'delay_ms': delay_s * 1000.0,
                'n_eligible': 0,
                'n_barred': 0,
                'n_trans': 0,
                'n_success': 0,
                'n_collision': 0,
                'n_miss': 0,
                'n_timeout': 0,
                'p_d': p_d,
                'snr_db': snr_db,
                'recent_delays': []
            }

        # 2. ACB check:
        # TS 36.331: draw u ~ U(0,1). If u <= p, pass; else barred.
        transmitting = []
        n_barred = 0
        t_acb_base_s = 4.0  # TS 36.331 default T_barring ~ 4s
        for dev in eligible:
            u = self.rng.random()
            if u <= acb_p:
                transmitting.append(dev)
            else:
                n_barred += 1
                # Barring backoff timer: (0.7 + 0.6 * rand) * t_acb
                t_bar = (0.7 + 0.6 * self.rng.random()) * t_acb_base_s
                dev.next_attempt_s = t_now_s + t_bar

        n_trans = len(transmitting)
        if n_trans == 0:
            return {
                't_s': t_now_s,
                'elevation_deg': elev,
                'delay_ms': delay_s * 1000.0,
                'n_eligible': n_eligible,
                'n_barred': n_barred,
                'n_trans': 0,
                'n_success': 0,
                'n_collision': 0,
                'n_miss': 0,
                'n_timeout': 0,
                'p_d': p_d,
                'snr_db': snr_db,
                'recent_delays': []
            }

        # 3. Preamble selection: uniform random in [0, n_preambles)
        preambles = self.rng.integers(0, self.n_preambles, size=n_trans)
        preamble_groups: Dict[int, List[DeviceState]] = {}
        for dev, pr in zip(transmitting, preambles):
            preamble_groups.setdefault(pr, []).append(dev)

        # 4. Mutually exclusive classification:
        n_success = 0
        n_collision = 0
        n_miss = 0
        n_timeout = 0
        recent_delays = []

        bi_s = float(bi_ms) / 1000.0
        # Preamble transmission duration in seconds (Format 0/1 symbol group duration * n_rep)
        # Format 0: 5.6 ms * n_rep
        t_tx_s = 0.0056 * float(n_rep)

        for pr, devs in preamble_groups.items():
            if len(devs) > 1:
                # Collision!
                n_collision += len(devs)
                for dev in devs:
                    dev.trans_count += 1
                    dev.last_failure_cause = 'collision'
                    if dev.trans_count >= self.preamble_trans_max:
                        dev.is_dropped = True
                    else:
                        t_bo = self.rng.uniform(0.0, bi_s)
                        dev.next_attempt_s = t_now_s + t_bo + t_tx_s
            else:
                # Single device cleanly transmitted preamble
                dev = devs[0]
                # Detection check
                u_detect = self.rng.random()
                if u_detect > p_d:
                    # Detection Miss!
                    n_miss += 1
                    dev.trans_count += 1
                    dev.last_failure_cause = 'miss'
                    if dev.trans_count >= self.preamble_trans_max:
                        dev.is_dropped = True
                    else:
                        t_bo = self.rng.uniform(0.0, bi_s)
                        dev.next_attempt_s = t_now_s + t_bo + t_tx_s
                else:
                    # Cleanly detected: Check round-trip time vs RAR window
                    # RTT = 2 * tau + t_proc
                    rtt_s = 2.0 * delay_s + self.processing_time_s
                    if rtt_s > rar_window_s:
                        # Timeout!
                        n_timeout += 1
                        dev.trans_count += 1
                        dev.last_failure_cause = 'timeout'
                        if dev.trans_count >= self.preamble_trans_max:
                            dev.is_dropped = True
                        else:
                            t_bo = self.rng.uniform(0.0, bi_s)
                            dev.next_attempt_s = t_now_s + t_bo + t_tx_s
                    else:
                        # Success!
                        n_success += 1
                        dev.trans_count += 1
                        dev.is_served = True
                        dev.access_delay_s = (t_now_s + rtt_s) - dev.arrival_time_s
                        recent_delays.append(dev.access_delay_s * 1000.0)

        # Sanity check: mutual exclusivity
        assert n_trans == (n_success + n_collision + n_miss + n_timeout), \
            f"Taxonomy violation: {n_trans} != {n_success} + {n_collision} + {n_miss} + {n_timeout}"

        return {
            't_s': t_now_s,
            'elevation_deg': elev,
            'delay_ms': delay_s * 1000.0,
            'n_eligible': n_eligible,
            'n_barred': n_barred,
            'n_trans': n_trans,
            'n_success': n_success,
            'n_collision': n_collision,
            'n_miss': n_miss,
            'n_timeout': n_timeout,
            'p_d': p_d,
            'snr_db': snr_db,
            'recent_delays': recent_delays
        }
