"""
nprach.py — Monte-Carlo NB-IoT NPRACH contention engine with failure taxonomy.

Implements:
1. Multi-phase and Beta(3,4) synchronized burst traffic distributed across
   the satellite pass visibility arc (Methodology §2.2 & Audit Checklist).
2. Step-by-step RAO contention loop over the LEO satellite pass.
3. ACB barring check (TS 36.331) against barring factor p.
4. Contention resolution over 48 contention subcarriers (TS 36.211).
5. 4-way mutually exclusive failure taxonomy (Methodology §2.3):
   - SUCCESS: unique preamble, cleanly detected, in-window.
   - COLLISION: >= 2 devices chose the same subcarrier.
   - DETECTION MISS: unique preamble cleanly sent but missed (1 - P_d).
   - TIMEOUT: unique and detected, but residual RTT > T_RAR.
6. Backoff Indicator (BI) retry scheduling (TS 36.321) up to preambleTransMax.
7. Tracking of access success within operational deadline (e.g. 45s).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.channel import elevation_to_snr_db, compute_detection_probability


class DeviceState:
    """Internal state tracking for an NB-IoT meter."""
    __slots__ = ('dev_id', 'arrival_time_s', 'next_attempt_s', 'trans_count',
                 'is_active', 'is_served', 'is_served_within_deadline',
                 'is_dropped', 'access_delay_s', 'last_failure_cause')

    def __init__(self, dev_id: int, arrival_time_s: float):
        self.dev_id = dev_id
        self.arrival_time_s = arrival_time_s
        self.next_attempt_s = arrival_time_s
        self.trans_count = 0
        self.is_active = False
        self.is_served = False
        self.is_served_within_deadline = False
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
        self.access_deadline_s = float(cfg.get('access_deadline_s', 45.0))

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

    def generate_traffic(self, n_devices: int, traffic_model: str = "multiphase") -> List[DeviceState]:
        """
        Generate activation arrival times for n_devices.
        
        Supports:
        - "multiphase": Distributed across early horizon, mid zenith, and late horizon
          to ensure all elevation bins are stimulated (resolving Fig. 2 U-shape).
        - "poisson": Uniform / Poisson continuous arrivals across the pass.
        - "beta": Single initial burst (3GPP TR 37.868 Beta(3,4)).
        """
        win_s = float(self.cfg.get('activation_window_s', 10.0))
        pass_dur = self.pass_duration_s

        if traffic_model == "multiphase":
            # 4 phases stimulating all 4 elevation bins: [10-25], [25-45], [45-70], [70-90]
            n_per = n_devices // 4
            rem = n_devices - 3 * n_per

            a = float(self.cfg.get('beta_a', 3))
            b = float(self.cfg.get('beta_b', 4))

            # Phase 1: Horizon rise (12°-17°, bin 10-25)
            arr1 = 20.0 + self.rng.beta(a, b, size=n_per) * 30.0
            # Phase 2: Moderate elevation (27°-35°, bin 25-45)
            arr2 = 115.0 + self.rng.beta(a, b, size=n_per) * 30.0
            # Phase 3: High elevation (48°-64°, bin 45-70)
            arr3 = 180.0 + self.rng.beta(a, b, size=n_per) * 30.0
            # Phase 4: Zenith overhead (75°-86°, bin 70-90)
            arr4 = 235.0 + self.rng.beta(a, b, size=rem) * 30.0

            arrivals = np.concatenate([arr1, arr2, arr3, arr4])
        elif traffic_model == "poisson":
            arrivals = self.rng.uniform(5.0, pass_dur * 0.9, size=n_devices)
        elif traffic_model == "beta":
            a = float(self.cfg.get('beta_a', 3))
            b = float(self.cfg.get('beta_b', 4))
            arrivals = self.rng.beta(a, b, size=n_devices) * win_s
        else:
            arrivals = self.rng.uniform(0.0, win_s, size=n_devices)

        arrivals.sort()
        devices = [DeviceState(dev_id=i, arrival_time_s=float(arrivals[i])) for i in range(n_devices)]
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
            elevation_deg=elev,
            altitude_km=self.cfg.get('altitude_km', 600.0),
            carrier_hz=self.cfg.get('carrier_hz', 2.0e9),
            p_tx_dbm=self.cfg.get('p_tx_dbm', 23.0),
            g_tx_dbi=self.cfg.get('g_tx_dbi', 0.0),
            g_rx_zenith_dbi=self.cfg.get('g_rx_zenith_dbi', 6.0),
            antenna_3db_beamwidth_deg=self.cfg.get('antenna_3db_beamwidth_deg', 65.0),
            noise_figure_db=self.cfg.get('noise_figure_db', 5.0),
            temperature_k=self.cfg.get('temperature_k', 290.0),
            scs_hz=self.scs_hz
        )
        p_d = compute_detection_probability(
            snr_db=snr_db,
            residual_doppler_hz=residual_doppler_hz,
            n_rep=n_rep,
            scs_hz=self.scs_hz,
            gamma_th_db=self.cfg.get('gamma_th_db', -1.0),
            k_slope=self.cfg.get('k_slope', 0.60)
        )

        # 1. Identify backlogged eligible devices:
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

        # 2. ACB check (TS 36.331)
        transmitting = []
        n_barred = 0
        t_acb_base_s = 4.0
        for dev in eligible:
            u = self.rng.random()
            if u <= acb_p:
                transmitting.append(dev)
            else:
                n_barred += 1
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

        # 3. Preamble subcarrier selection: uniform random in [0, n_preambles)
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
        # Preamble transmission duration in seconds (Format 0/1: 5.6 ms * n_rep)
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
                # Unique preamble cleanly sent
                dev = devs[0]
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
                    # In NTN Rel-17, UE pre-compensates common delay; residual RTT is checked vs window
                    # For legacy terrestrial RAR (e.g. 8ms uncompensated), RTT = 2*delay + t_proc exceeds window
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
                        dev.is_served_within_deadline = (dev.access_delay_s <= self.access_deadline_s)
                        recent_delays.append(dev.access_delay_s * 1000.0)

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
