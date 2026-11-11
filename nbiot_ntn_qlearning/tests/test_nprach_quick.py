import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.orbit import generate_pass_trajectory_from_config
from src.nprach import NPRACHSimulator

cfg = load_config()
traj = generate_pass_trajectory_from_config(cfg)

# Test 1: Tiny N (e.g. 10 devices) -> P_s should be ~1
sim_tiny = NPRACHSimulator(cfg, traj, seed=200)
devices_tiny = sim_tiny.generate_traffic(10, traffic_model="beta")
rao_step_s = 0.160
t = 0.0
while t < traj['duration_s']:
    sim_tiny.simulate_rao(t, devices_tiny, acb_p=1.0, bi_ms=1024, n_rep=8,
                          residual_doppler_hz=10.0, rar_window_s=0.040)
    t += rao_step_s

served_tiny = sum(1 for d in devices_tiny if d.is_served)
p_s_tiny = served_tiny / 10.0
print(f"Test 1 (Tiny N=10): Served = {served_tiny}/10 -> P_s = {p_s_tiny:.2f} (Expected ~1.0)")

# Test 2: Large N (e.g. 1000 devices) with ACB off (p=1.0) and short BI (256ms)
# Expect collision collapse
sim_heavy = NPRACHSimulator(cfg, traj, seed=200)
devices_heavy = sim_heavy.generate_traffic(1000, traffic_model="beta")
t = 0.0
total_coll = 0
total_trans = 0
while t < 25.0:  # first 25s of heavy sync burst
    out = sim_heavy.simulate_rao(t, devices_heavy, acb_p=1.0, bi_ms=256, n_rep=2,
                                 residual_doppler_hz=10.0, rar_window_s=0.040)
    total_coll += out['n_collision']
    total_trans += out['n_trans']
    t += rao_step_s

print(f"Test 2 (Heavy burst N=1000, ACB off): Total trans = {total_trans}, Collisions = {total_coll} ({total_coll/max(1, total_trans)*100:.1f}%)")

# Test 3: Terrestrial RAR window (8 ms) vs NTN delay
# When 2*tau + 3ms > 8ms, timeouts should occur!
sim_to = NPRACHSimulator(cfg, traj, seed=200)
devices_to = sim_to.generate_traffic(50, traffic_model="beta")
t = 0.0 # near horizon elevation 10 deg, delay ~ 6.3 ms -> RTT = 12.6 + 3 = 15.6 ms > 8 ms!
timeouts = 0
while t < 10.0:
    out = sim_to.simulate_rao(t, devices_to, acb_p=1.0, bi_ms=1024, n_rep=32,
                              residual_doppler_hz=10.0, rar_window_s=0.008) # 8ms terrestrial RAR
    timeouts += out['n_timeout']
    t += rao_step_s
print(f"Test 3 (Terrestrial 8ms RAR at pass edge): Timeouts observed = {timeouts} (Expected > 0)")
