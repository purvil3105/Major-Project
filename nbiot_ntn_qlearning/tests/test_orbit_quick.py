import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.orbit import generate_pass_trajectory_from_config

cfg = load_config()
traj = generate_pass_trajectory_from_config(cfg)
print(f"Duration: {traj['duration_s']:.1f} s ({traj['duration_s']/60:.2f} min)")
print(f"Max Elev: {traj['max_elevation_deg']:.2f} deg")
print(f"Delay range: {traj['delay_ms'].min():.2f} - {traj['delay_ms'].max():.2f} ms")
print(f"Doppler range: {traj['raw_doppler_hz'].min()/1e3:.2f} to {traj['raw_doppler_hz'].max()/1e3:.2f} kHz")
