"""
nprach.py — vectorised Monte-Carlo NB-IoT NPRACH simulator over one LEO pass.

Per decision epoch (one base RAO period, default 320 ms):
  1. New RA initiations pass an access-barring check (AB bitmap modelled as probability p, checked
     only at initiation, not on preamble retransmissions). Barred meters wait (0.7+0.6U) T_bar.
  2. A meter that passes selects CE0/CE1/CE2 from its measured SNR against the thresholds the
     network broadcasts for the active repetition profile (threshold_k = SNR99(N_k)).
  3. Each CE level k has its own subcarrier pool R_k and periodicity m_k * T_epoch >= N_k x 6.4 ms.
     Meters pick a subcarrier uniformly at random.
  4. Ground-truth outcome per transmitting meter (mutually exclusive):
       collision  >= 2 meters on the same subcarrier (all fail at Msg4; no capture effect)
       miss       alone on the subcarrier, not detected (prob 1 - P_d(SNR_i, N_k))
       timeout    alone and detected, but a GNSS-degraded meter's timing error exceeds the CP,
                  so Msg3 is not received in its grant
       success    otherwise
  5. Failures: back off U(0, BI); after max_attempts_per_ce at a level the meter moves up one CE
     level; after preamble_trans_max attempts it is dropped.
  6. What the gNB can observe is returned separately: per pool, the number of idle-looking
     subcarriers, successes, collided subcarriers flagged by an early collision detector, and
     Msg3 failures. The controller sees only these.
"""

import numpy as np

from src.channel import detection_probability, outdoor_snr_db, snr99_db, timeout_probability


class NPRACHSimulator:
    def __init__(self, cfg, traj, n_devices, seed_seq, coupling_loss_mean_db=None,
                 gnss_degraded_frac=None, regime=None, rao_period_ms=None, terrestrial=False):
        self.cfg = cfg
        self.traj = traj
        self.regime = regime or cfg["detection_regime"]
        self.terrestrial = terrestrial
        self.epoch_s = float(rao_period_ms or cfg["rao_period_ms"]) / 1000.0
        self.R = np.asarray(cfg["ce_subcarriers"], dtype=int)
        self.n_ce = len(self.R)
        self.rep_ms = float(cfg["preamble_rep_ms"])
        self.t_proc_s = float(cfg["processing_time_ms"]) / 1000.0
        self.max_att_ce = int(cfg["max_attempts_per_ce"])
        self.max_att = int(cfg["preamble_trans_max"])
        self.p_cd = float(cfg["ecd_detect_prob"])
        self.duration_s = float(traj["duration_s"])
        self.n_epochs = int(np.floor(self.duration_s / self.epoch_s)) + 1

        ss_traffic, ss_sim = np.random.SeedSequence(seed_seq).spawn(2)
        rt = np.random.default_rng(ss_traffic)   # common random numbers across policies
        self.rng = np.random.default_rng(ss_sim)

        mu = cfg["coupling_loss_mean_db"] if coupling_loss_mean_db is None else coupling_loss_mean_db
        fdeg = cfg["gnss_degraded_frac"] if gnss_degraded_frac is None else gnss_degraded_frac
        n = int(n_devices)
        self.n = n
        # Traffic: Poisson-like periodic reports over the pass + one Beta(3,4) burst at a random time
        n_burst = int(round(cfg["burst_fraction"] * n))
        t_regular = rt.uniform(0.0, self.duration_s, size=n - n_burst)
        t_b0 = rt.uniform(0.0, max(1.0, self.duration_s - cfg["burst_window_s"]))
        t_burst = t_b0 + rt.beta(cfg["beta_a"], cfg["beta_b"], size=n_burst) * cfg["burst_window_s"]
        self.arrival = np.concatenate([t_regular, t_burst])
        self.loss_db = rt.normal(mu, cfg["coupling_loss_std_db"], size=n)
        self.degraded = (rt.random(n) < fdeg) & (not terrestrial)
        self.burst_start_s = t_b0

        # Device state
        self.next_t = self.arrival.copy()
        self.initiated = np.zeros(n, bool)
        self.ce = np.zeros(n, int)
        self.att_ce = np.zeros(n, int)
        self.att = np.zeros(n, int)
        self.served = np.zeros(n, bool)
        self.dropped = np.zeros(n, bool)
        self.t_success = np.full(n, np.nan)
        self.airtime_ms = np.zeros(n)

        # Per-epoch geometry, precomputed once (vectorised)
        te = np.minimum(np.arange(self.n_epochs) * self.epoch_s, self.duration_s)
        tr = traj
        self._elev = np.interp(te, tr["time_s"], tr["elevation_deg"])
        if terrestrial:
            self._delay = np.zeros_like(te)
            self._snr = np.full_like(te, float(cfg["terrestrial_snr_db"]))
        else:
            self._delay = np.interp(te, tr["time_s"], tr["delay_s"])
            rng_km = np.interp(te, tr["time_s"], tr["slant_range_km"])
            self._snr = outdoor_snr_db(rng_km, self._elev, tr.get("altitude_km", 600.0), cfg)
        self._p_to = timeout_probability(self._elev, cfg) if not terrestrial else np.zeros_like(te)
        self._cache = {}

    # ------------------------------------------------------------------ geometry
    def geometry_at_epoch(self, e):
        """(elevation_deg, one-way delay_s, outdoor SNR dB) at epoch index e."""
        return float(self._elev[e]), float(self._delay[e]), float(self._snr[e])

    def _profile_info(self, reps):
        key = tuple(int(r) for r in reps)
        if key not in self._cache:
            r = np.asarray(key, int)
            self._cache[key] = (r, self.period_multipliers(r), snr99_db(r, self.cfg, self.regime))
        return self._cache[key]

    def period_multipliers(self, reps):
        need = np.asarray(reps) * self.rep_ms / 1000.0
        out = []
        for x in need:
            m = next((m for m in self.cfg["period_multipliers"] if m * self.epoch_s >= x),
                     self.cfg["period_multipliers"][-1])
            out.append(m)
        return np.asarray(out, int)

    def backlog(self, t):
        return int(np.sum((self.arrival <= t) & ~self.served & ~self.dropped))

    # ------------------------------------------------------------------ one epoch
    def step(self, epoch_idx, acb_p, bi_ms, reps):
        cfg, rng = self.cfg, self.rng
        t = epoch_idx * self.epoch_s
        elev, delay_s, snr_out = self.geometry_at_epoch(epoch_idx)
        reps, mult, thresholds = self._profile_info(reps)
        active = (epoch_idx % mult) == 0

        pending = (~self.served) & (~self.dropped) & (self.arrival <= t) & (self.next_t <= t + 1e-9)

        # 1. Access barring at initiation only
        new = pending & ~self.initiated
        idx_new = np.flatnonzero(new)
        if idx_new.size:
            u = rng.random(idx_new.size)
            ok = u <= acb_p
            barred = idx_new[~ok]
            self.next_t[barred] = t + (0.7 + 0.6 * rng.random(barred.size)) * cfg["barring_time_s"]
            start = idx_new[ok]
            self.initiated[start] = True
            meas = snr_out - self.loss_db[start] + rng.normal(0, cfg["rsrp_meas_std_db"], start.size)
            ce = np.full(start.size, self.n_ce - 1)
            for k in range(self.n_ce - 2, -1, -1):
                ce[meas >= thresholds[k]] = k
            self.ce[start] = ce
            self.att_ce[start] = 0

        tx_all = pending & self.initiated
        res = {"t_s": t, "elevation_deg": elev, "delay_ms": delay_s * 1e3, "snr_out_db": snr_out,
               "acb_p": acb_p, "bi_ms": bi_ms, "rep_profile": tuple(int(r) for r in reps),
               "n_pending": int(pending.sum()),
               "n_success": 0, "n_collision": 0, "n_miss": 0, "n_timeout": 0, "n_trans": 0,
               "airtime_ms": 0.0, "pools": []}
        bi_s = float(bi_ms) / 1000.0
        rtt = 2.0 * delay_s

        for k in range(self.n_ce):
            pool = {"k": k, "R": int(self.R[k]), "active": bool(active[k]), "reps": int(reps[k]),
                    "idle_obs": 0, "succ": 0, "flagged": 0, "msg3_fail": 0}
            if not active[k]:
                res["pools"].append(pool)
                continue
            idx = np.flatnonzero(tx_all & (self.ce == k))
            pool["idle_obs"] = int(self.R[k])
            if idx.size == 0:
                res["pools"].append(pool)
                continue
            n_k = idx.size
            t_pre = reps[k] * self.rep_ms / 1000.0
            self.airtime_ms[idx] += reps[k] * self.rep_ms
            res["airtime_ms"] += n_k * reps[k] * self.rep_ms
            sc = rng.integers(0, self.R[k], size=n_k)
            counts = np.bincount(sc, minlength=self.R[k])
            pd = detection_probability(snr_out - self.loss_db[idx], reps[k], cfg, self.regime)
            single = counts[sc] == 1
            detected = rng.random(n_k) < pd

            # collisions: energy on a collided subcarrier is detected if any member is detected
            q_none = np.bincount(sc, weights=np.log(np.maximum(1.0 - pd, 1e-300)), minlength=self.R[k])
            coll_sc = np.flatnonzero(counts >= 2)
            coll_det = rng.random(coll_sc.size) < (1.0 - np.exp(q_none[coll_sc]))
            flagged = coll_det & (rng.random(coll_sc.size) < self.p_cd)

            s_det = single & detected
            miss = single & ~detected
            p_to = float(self._p_to[epoch_idx])
            timeout = s_det & self.degraded[idx] & (rng.random(n_k) < p_to)
            success = s_det & ~timeout
            collision = ~single

            # ground truth bookkeeping
            ok_idx = idx[success]
            self.served[ok_idx] = True
            self.t_success[ok_idx] = t + t_pre + 2.0 * rtt + 2.0 * self.t_proc_s
            self.att[ok_idx] += 1
            fail_idx = idx[~success]
            self.att[fail_idx] += 1
            self.att_ce[fail_idx] += 1
            to_idx = idx[timeout]
            refix = to_idx[rng.random(to_idx.size) < cfg["gnss_refix_prob"]]
            self.degraded[refix] = False
            drop = fail_idx[self.att[fail_idx] >= self.max_att]
            self.dropped[drop] = True
            keep = fail_idx[self.att[fail_idx] < self.max_att]
            up = keep[(self.att_ce[keep] >= self.max_att_ce) & (self.ce[keep] < self.n_ce - 1)]
            self.ce[up] += 1
            self.att_ce[up] = 0
            self.next_t[keep] = t + t_pre + rtt + self.t_proc_s + rng.uniform(0.0, bi_s, keep.size)

            n_s, n_c, n_m, n_t = int(success.sum()), int(collision.sum()), int(miss.sum()), int(timeout.sum())
            assert n_s + n_c + n_m + n_t == n_k
            res["n_success"] += n_s
            res["n_collision"] += n_c
            res["n_miss"] += n_m
            res["n_timeout"] += n_t
            res["n_trans"] += n_k

            # gNB observables
            n_coll_det = int(coll_det.sum())
            pool["idle_obs"] = int(self.R[k] - s_det.sum() - n_coll_det)
            pool["succ"] = n_s
            pool["flagged"] = int(flagged.sum())
            pool["msg3_fail"] = n_t + int(n_coll_det - flagged.sum())   # unflagged collisions also fail Msg3
            pool["n_tx_true"] = n_k
            res["pools"].append(pool)
        return res

    # ------------------------------------------------------------------ summary
    def summary(self):
        delays = self.t_success - self.arrival
        served = self.served
        d = delays[served]
        dl = float(self.cfg["deadline_s"])
        n_ok = int(served.sum())
        return {
            "p_s": n_ok / self.n,
            "p_s_deadline": float(np.sum(served & (delays <= dl))) / self.n,
            "p_drop": float(self.dropped.sum()) / self.n,
            "mean_delay_s": float(np.mean(d)) if d.size else np.nan,
            "p95_delay_s": float(np.percentile(d, 95)) if d.size else np.nan,
            "airtime_per_success_ms": float(self.airtime_ms.sum()) / max(1, n_ok),
            "attempts_per_success": float(self.att.sum()) / max(1, n_ok),
        }, d
