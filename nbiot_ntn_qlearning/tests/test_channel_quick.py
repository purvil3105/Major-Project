import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.channel import compute_detection_probability, compute_miss_probability, elevation_to_snr_db

# Check Anchor 1
p_miss_1 = compute_miss_probability(snr_db=8.0, residual_doppler_hz=10.0, n_rep=64)
p_d_1 = compute_detection_probability(snr_db=8.0, residual_doppler_hz=10.0, n_rep=64)
print(f"Anchor 1 (8 dB, 64 rep, 10 Hz): P_d = {p_d_1:.4f}, P_miss = {p_miss_1:.4e} (target <= 1e-2)")

# Check Anchor 2
p_miss_2 = compute_miss_probability(snr_db=15.0, residual_doppler_hz=10.0, n_rep=32)
p_d_2 = compute_detection_probability(snr_db=15.0, residual_doppler_hz=10.0, n_rep=32)
print(f"Anchor 2 (15 dB, 32 rep, 10 Hz): P_d = {p_d_2:.4f}, P_miss = {p_miss_2:.4e} (target <= 1e-2)")

# Check monotonicity with N_rep at edge SNR (-3 dB)
for nrep in [2, 8, 32]:
    pd = compute_detection_probability(snr_db=-3.0, residual_doppler_hz=10.0, n_rep=nrep)
    print(f"Edge SNR (-3 dB), N_rep = {nrep:2d}: P_d = {pd:.4f}")

# Check Doppler stress degradation (250 Hz vs 10 Hz)
pd_nom = compute_detection_probability(snr_db=2.0, residual_doppler_hz=10.0, n_rep=8)
pd_stress = compute_detection_probability(snr_db=2.0, residual_doppler_hz=250.0, n_rep=8)
print(f"Doppler impact (SNR=2dB, N_rep=8): P_d(10Hz) = {pd_nom:.4f} vs P_d(250Hz) = {pd_stress:.4f}")
