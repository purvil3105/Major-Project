"""
experiment.py — pre-training, baselines search, main evaluation and sweeps.

Seeding: every run gets SeedSequence([seed, pass_index, stage_tag]). The simulator derives a
traffic stream (arrivals, coupling losses, GNSS-degraded flags) and a separate simulation
stream from it, so all policies see identical traffic on the same (pass, seed): paired design.
Agents draw exploration noise from their own stream. Two runs with the same config are
byte-identical.
"""

import json
import os
import platform
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.agent import TabularQLearningAgent
from src.baselines import B1_ACTION, B2_ACTION, heuristic_action
from src.config_loader import config_hash, load_config, resolve_path
from src.mdp import RACH_MDP
from src.nprach import NPRACHSimulator
from src.orbit import get_passes

LEARNING_VARIANTS = {
    # name: (use_cause_state, use_satellite_state, lumped_reward, terrestrial_training)
    "Proposed": (True, True, False, False),
    "Abl_LumpedReward": (True, True, True, False),
    "Abl_NoCause": (False, True, True, False),
    "B3_TerrestrialFrozen": (True, False, False, True),
}
POLICIES = ["Proposed", "Proposed_Cold", "Abl_LumpedReward", "Abl_NoCause",
            "B1_FixedTerrestrial", "B2_StaticNTN", "B3_TerrestrialFrozen", "B4_Heuristic", "B5_BestFixed"]
ELEV_BINS = [10, 20, 35, 50, 70, 90]

_G = {}


def _passes(cfg, role, altitude_km=None):
    key = (role, altitude_km)
    if key not in _G.setdefault("passes", {}):
        limit = None if role == "test" else 0
        _G["passes"][key] = get_passes(cfg, role, altitude_km=altitude_km, limit=limit)
    return _G["passes"][key]


def _variant_of(policy):
    return "Proposed" if policy == "Proposed_Cold" else policy


# ---------------------------------------------------------------------------- one pass
def run_episode(cfg, traj, policy, n_devices, seed_seq, q_tables=None, b5_action=None,
                overrides=None, log_rao=False, terrestrial=False, agent=None, mdp=None):
    """Simulate one pass under `policy`. Returns (summary, rao_log, delays, elevation_bins)."""
    ov = overrides or {}
    sim = NPRACHSimulator(cfg, traj, n_devices, seed_seq,
                          coupling_loss_mean_db=ov.get("coupling_loss_mean_db"),
                          gnss_degraded_frac=ov.get("gnss_degraded_frac"),
                          regime=ov.get("detection_regime"),
                          rao_period_ms=ov.get("rao_period_ms"), terrestrial=terrestrial)
    learning = policy in ("Proposed", "Proposed_Cold", "Abl_LumpedReward", "Abl_NoCause")
    variant = _variant_of(policy)
    if mdp is None:
        cs, ss, lumped, _ = LEARNING_VARIANTS.get(variant, (True, True, False, False))
        mdp = RACH_MDP(cfg, use_cause_state=cs, use_satellite_state=ss, lumped_reward=lumped)
    mdp.reset()
    if agent is None and (learning or policy == "B3_TerrestrialFrozen"):
        rng_agent = np.random.default_rng(np.random.SeedSequence(list(seed_seq) + [7]))
        if policy == "Proposed_Cold":
            agent = TabularQLearningAgent(mdp.n_states, mdp.n_actions, cfg["alpha"], cfg["gamma"],
                                          cfg["eps0"], cfg["eps_min"], cfg["eps_decay"], rng_agent)
        else:
            eps = cfg["eps_warm"] if learning else 0.0
            agent = TabularQLearningAgent(mdp.n_states, mdp.n_actions, cfg["alpha"], cfg["gamma"],
                                          eps, cfg["eps_min"] if learning else 0.0, cfg["eps_decay"],
                                          rng_agent, q_table=q_tables[variant])
    greedy_only = policy == "B3_TerrestrialFrozen"

    log, prev = [], None
    tot = {"n_trans": 0, "n_success": 0, "n_collision": 0, "n_miss": 0, "n_timeout": 0}
    elev_att = np.zeros(len(ELEV_BINS) - 1)
    elev_succ = np.zeros(len(ELEV_BINS) - 1)
    for e in range(sim.n_epochs):
        t = e * sim.epoch_s
        elev, _, _ = sim.geometry_at_epoch(e)
        prog = t / max(sim.duration_s, 1.0)
        s = mdp.encode_state(elev, prog)
        if agent is not None:
            a = agent.select_action(s, greedy=greedy_only)
            p, bi, reps = mdp.decode_action(a)
        else:
            if policy == "B1_FixedTerrestrial":
                act = B1_ACTION
            elif policy == "B2_StaticNTN":
                act = B2_ACTION
            elif policy == "B4_Heuristic":
                load = mdp.estimates()[0] / float(sum(cfg["ce_subcarriers"]))
                act = heuristic_action(elev, load)
            elif policy == "B5_BestFixed":
                act = b5_action
            elif policy.startswith("Fixed_"):
                act = ov["fixed_action"]
            else:
                raise ValueError(policy)
            p, bi, reps = act[0], act[1], cfg["rep_profiles"][act[2]]
            a = None
        rao = sim.step(e, p, bi, reps)
        for kk in tot:
            tot[kk] += rao[kk]
        r = mdp.observe(rao)
        informative = r is not None and any(
            pl["active"] and (pl["succ"] + pl["flagged"] + pl["msg3_fail"] > 0 or pl["idle_obs"] < pl["R"])
            for pl in rao["pools"])
        if agent is not None and not greedy_only:
            if prev is not None and informative:
                agent.update(prev[0], prev[1], prev[2], s)
            if informative:
                prev = (s, a, r)
        if rao["n_trans"]:
            b = min(max(np.searchsorted(ELEV_BINS, elev, side="right") - 1, 0), len(ELEV_BINS) - 2)
            elev_att[b] += rao["n_trans"]
            elev_succ[b] += rao["n_success"]
        if log_rao:
            est = mdp.estimates()
            log.append({"step": e, "t_s": t, "elevation_deg": elev, "acb_p": p, "bi_ms": bi,
                         "rep_ce2": int(reps[-1]), "rep_ce0": int(reps[0]), "n_trans": rao["n_trans"],
                         "n_success": rao["n_success"], "n_collision": rao["n_collision"],
                         "n_miss": rao["n_miss"], "n_timeout": rao["n_timeout"],
                         "n_pending": rao["n_pending"], "reward": np.nan if r is None else r,
                         "est_attempts": est[0], "est_coll": est[1], "est_miss": est[2], "est_timeout": est[3]})
    if agent is not None and not greedy_only and prev is not None:
        agent.update(prev[0], prev[1], prev[2], prev[0], done=True)
    summ, delays = sim.summary()
    nt = max(1, tot["n_trans"])
    summ.update({"policy": policy, "n_devices": n_devices, "duration_s": sim.duration_s,
                 "peak_elev_deg": traj["max_elevation_deg"], **tot,
                 "collision_rate": tot["n_collision"] / nt, "miss_rate": tot["n_miss"] / nt,
                 "timeout_rate": tot["n_timeout"] / nt})
    return summ, log, delays, (elev_att, elev_succ), agent


# ---------------------------------------------------------------------------- pre-training
def pretrain(cfg, variant, out_path):
    """Warm-start table for a learning variant, trained only on passes of the training satellites."""
    cs, ss, lumped, terrestrial = LEARNING_VARIANTS[variant]
    mdp = RACH_MDP(cfg, use_cause_state=cs, use_satellite_state=ss, lumped_reward=lumped)
    rng = np.random.default_rng(np.random.SeedSequence([cfg["pretrain_seed"], list(LEARNING_VARIANTS).index(variant)]))
    agent = TabularQLearningAgent(mdp.n_states, mdp.n_actions, cfg["alpha"], cfg["gamma"],
                                  cfg["eps0"], cfg["eps_min"], cfg["pretrain_eps_decay"], rng)
    train = _passes(cfg, "train")
    hist = []
    for i in range(int(cfg["pretrain_passes"])):
        traj = train[i % len(train)]
        n = int(rng.choice(cfg["device_sweep"]))
        ov = {"coupling_loss_mean_db": float(rng.choice(cfg["coupling_loss_sweep_db"])),
              "gnss_degraded_frac": float(rng.choice(cfg["gnss_degraded_sweep"]))}
        summ, _, _, _, agent = run_episode(cfg, traj, "Proposed", n, [cfg["pretrain_seed"], list(LEARNING_VARIANTS).index(variant), i],
                                           overrides=ov, terrestrial=terrestrial, agent=agent, mdp=mdp)
        hist.append({"variant": variant, "pass": i, "n_devices": n, **ov,
                     "p_s_deadline": summ["p_s_deadline"], "eps": agent.eps})
    agent.save(out_path)
    return hist


# ---------------------------------------------------------------------------- workers
def _init_worker(cfg_path, qdir):
    _G["cfg"] = load_config(cfg_path)
    _G["q"] = {v: np.load(os.path.join(qdir, f"q_{v}.npy")) for v in LEARNING_VARIANTS
               if os.path.exists(os.path.join(qdir, f"q_{v}.npy"))}


def _task(task):
    cfg = _G["cfg"]
    traj = _passes(cfg, "test", task.get("altitude_km"))[task["pass_idx"]]
    ov = dict(task.get("overrides", {}))
    if task.get("fixed_action") is not None:
        ov["fixed_action"] = tuple(task["fixed_action"])
    summ, log, delays, (ea, es), _ = run_episode(
        cfg, traj, task["policy"], task["n_devices"], [task["seed"], task["pass_idx"], task["tag"]],
        q_tables=_G["q"], b5_action=task.get("b5_action"), overrides=ov, log_rao=task.get("log", False))
    summ.update({"seed": task["seed"], "pass_idx": task["pass_idx"], "experiment": task["experiment"],
                 **{k: v for k, v in task.get("overrides", {}).items()},
                 "altitude_km": task.get("altitude_km") or round(traj["altitude_km"], 1)})
    if task.get("fixed_action") is not None:
        summ["action"] = str(tuple(task["fixed_action"]))
    meta = {k: summ[k] for k in ("policy", "seed", "pass_idx", "experiment")}
    log = [{**meta, **row} for row in log]
    eb = [{**meta, "elev_lo": ELEV_BINS[i], "elev_hi": ELEV_BINS[i + 1], "attempts": ea[i], "successes": es[i]}
          for i in range(len(ea))]
    dl = delays if task.get("keep_delays") else np.array([])
    return summ, log, eb, dl


def _pool_map(tasks, cfg_path, qdir, n_workers, desc):
    from tqdm import tqdm
    if n_workers <= 1:
        _init_worker(cfg_path, qdir)
        return [_task(t) for t in tqdm(tasks, desc=desc)]
    with Pool(n_workers, initializer=_init_worker, initargs=(cfg_path, qdir)) as pool:
        return list(tqdm(pool.imap(_task, tasks, chunksize=2), total=len(tasks), desc=desc))


def _pretrain_task(args):
    cfg_path, variant, path = args
    return pretrain(load_config(cfg_path), variant, path)


# ---------------------------------------------------------------------------- driver
def run_all_experiments(cfg_path=None, output_dir="results", quick=False):
    cfg = load_config(cfg_path)
    cfg_path = resolve_path(cfg_path or os.path.join("config", "default.yaml"))
    out = resolve_path(output_dir)
    qdir = os.path.join(out, "qtables")
    os.makedirs(qdir, exist_ok=True)
    nw = int(cfg.get("n_workers") or max(1, (os.cpu_count() or 2) - 1))
    t_start = time.time()
    seeds = cfg["seeds"][:2] if quick else cfg["seeds"]
    n_test = len(_passes(cfg, "test"))
    pass_ids = list(range(n_test))[: (3 if quick else n_test)]
    sweep_ids = [int(round(x)) for x in np.linspace(0, n_test - 1, int(cfg["sweep_passes"]))]
    sweep_seeds = cfg["sweep_seeds"][:2] if quick else cfg["sweep_seeds"]

    # 1. warm-start tables (training satellites only)
    print("[1/5] Pre-training warm-start tables on training-satellite passes ...")
    if quick:
        cfg["pretrain_passes"] = 3
    jobs = [(cfg_path, v, os.path.join(qdir, f"q_{v}.npy")) for v in LEARNING_VARIANTS]
    if quick:
        hists = [pretrain(cfg, v, p) for _, v, p in jobs]
    else:
        with Pool(min(nw, len(jobs))) as pool:
            hists = pool.map(_pretrain_task, jobs)
    pd.DataFrame([h for hh in hists for h in hh]).to_csv(os.path.join(out, "pretrain_history.csv"), index=False)

    # 2. B5: exhaustive search over all fixed actions
    print("[2/5] B5 exhaustive search over fixed configurations ...")
    mdp = RACH_MDP(cfg)
    b5_seeds = cfg["b5_search_seeds"][:1] if quick else cfg["b5_search_seeds"]
    tasks = [{"policy": "Fixed_search", "fixed_action": act, "pass_idx": pi, "seed": sd, "tag": 1,
              "n_devices": cfg["n_devices"], "experiment": "b5_search"}
             for act in mdp.actions for pi in pass_ids for sd in b5_seeds]
    res = _pool_map(tasks, cfg_path, qdir, nw, "B5 search")
    b5 = pd.DataFrame([r[0] for r in res])
    b5.to_csv(os.path.join(out, "b5_search.csv"), index=False)
    rank = b5.groupby("action")["p_s_deadline"].mean().sort_values(ascending=False)
    b5_action = tuple(eval(rank.index[0]))  # noqa: S307 (our own tuple repr)
    print(f"      best fixed action {b5_action}: mean P_s(deadline) = {rank.iloc[0]:.3f}")

    # 3. main evaluation
    print("[3/5] Main evaluation ...")
    log_seeds = set(seeds[:3])
    tasks = [{"policy": pol, "pass_idx": pi, "seed": sd, "tag": 1, "n_devices": cfg["n_devices"],
              "experiment": "main", "b5_action": b5_action, "log": sd in log_seeds, "keep_delays": True}
             for pol in POLICIES for pi in pass_ids for sd in seeds]
    res = _pool_map(tasks, cfg_path, qdir, nw, "Main")
    pd.DataFrame([r[0] for r in res]).to_csv(os.path.join(out, "metrics.csv"), index=False)
    pd.DataFrame([row for r in res for row in r[1]]).to_csv(os.path.join(out, "rao_log.csv.gz"), index=False)
    pd.DataFrame([row for r in res for row in r[2]]).to_csv(os.path.join(out, "elevation_bins.csv"), index=False)
    dl = [pd.DataFrame({"policy": r[0]["policy"], "seed": r[0]["seed"], "pass_idx": r[0]["pass_idx"],
                        "delay_s": r[3]}) for r in res]
    pd.concat(dl).to_csv(os.path.join(out, "delays.csv.gz"), index=False)

    # 4. sweeps
    print("[4/5] Sensitivity sweeps ...")
    tasks = []

    def add(exp, policies, overrides=None, n=None, altitude=None, ids=None):
        for pol in policies:
            for pi in (ids or sweep_ids):
                for sd in sweep_seeds:
                    tasks.append({"policy": pol, "pass_idx": pi, "seed": sd, "tag": 2, "experiment": exp,
                                  "n_devices": n or cfg["n_devices"], "overrides": overrides or {},
                                  "b5_action": b5_action, "altitude_km": altitude})
    core = ["Proposed", "B2_StaticNTN", "B4_Heuristic", "B5_BestFixed"]
    for n in cfg["device_sweep"]:
        for mu in cfg["coupling_loss_sweep_db"]:
            add("load_x_coupling", core, {"coupling_loss_mean_db": mu}, n=n)
    add("regime_gnss_free", core, {"detection_regime": "gnss_free"})
    for f in cfg["gnss_degraded_sweep"]:
        add("gnss_degraded", ["Proposed", "Abl_NoCause", "B2_StaticNTN", "B5_BestFixed"], {"gnss_degraded_frac": f})
    for rp in cfg["rao_period_sweep_ms"]:
        add("rao_period", ["Proposed", "B2_StaticNTN", "B4_Heuristic"], {"rao_period_ms": rp})
    for alt in cfg["synthetic_altitudes_km"]:
        n_alt = len(_passes(cfg, "test", float(alt)))
        ids = [int(round(x)) for x in np.linspace(0, n_alt - 1, int(cfg["sweep_passes"]))]
        add("altitude", ["Proposed", "B2_StaticNTN", "B4_Heuristic"], {}, altitude=float(alt), ids=ids)
    res = _pool_map(tasks, cfg_path, qdir, nw, "Sweeps")
    pd.DataFrame([r[0] for r in res]).to_csv(os.path.join(out, "sensitivity.csv"), index=False)

    # 5. manifest
    import importlib.metadata as md
    manifest = {
        "config_hash": config_hash(cfg), "b5_action": b5_action, "seeds": seeds, "test_passes": [
            {"idx": i, "rise_utc": tr["rise_utc"], "peak_deg": round(tr["max_elevation_deg"], 2),
             "duration_s": tr["duration_s"]} for i, tr in enumerate(_passes(cfg, "test"))],
        "n_train_passes": len(_passes(cfg, "train")), "python": platform.python_version(),
        "packages": {p: md.version(p) for p in ["numpy", "scipy", "pandas", "skyfield", "sgp4", "matplotlib"]},
        "runtime_s": round(time.time() - t_start, 1), "quick": quick,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[5/5] Done in {manifest['runtime_s']} s. Results in {out}")
    return manifest
