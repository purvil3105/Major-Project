"""
baselines.py — non-learning reference controllers.

B1  Fixed terrestrial-style configuration: no barring, short backoff, lowest repetition profile.
B2  Static NTN-aware configuration (renamed from "Rel-17 default": 3GPP defines no such preset).
B4  Hand-written channel-aware heuristic using only observable inputs (elevation from
    ephemeris, load estimated from NPRACH observations).
B5  Best fixed configuration: the single (p, BI, profile) with the highest mean deadline
    success over the test passes, found by exhaustive search (a genie: it sees the test set).
B3 (terrestrial-frozen Q-table) is a learning-based baseline and lives in experiment.py.
"""

B1_ACTION = (1.0, 256, 0)
B2_ACTION = (0.7, 1024, 1)


def heuristic_action(elevation_deg, load_per_subcarrier):
    """B4: repetition profile from elevation, barring/backoff from the estimated load."""
    if elevation_deg < 25.0:
        prof = 2
    elif elevation_deg < 50.0:
        prof = 1
    else:
        prof = 0
    if load_per_subcarrier > 1.5:
        p, bi = 0.3, 4096
    elif load_per_subcarrier > 0.8:
        p, bi = 0.5, 1024
    elif load_per_subcarrier > 0.3:
        p, bi = 0.7, 1024
    else:
        p, bi = 1.0, 256
    return p, bi, prof
