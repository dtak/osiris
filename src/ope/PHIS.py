"""
Implementation of per-horizon importance sampling
"""

# %% Imports
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp

## Per-horizon WIS
def PHWIS(history, verbose=False):
    if verbose:
        raise NotImplementedError()
    traj_lens_unique, traj_lens_counts = np.unique(history.traj_lens, return_counts=True)
    W_ell = traj_lens_counts / len(history.traj_lens)
    log_rho = np.array([np.log(p).sum() for p in history.rho_ts]) # time
    estimate = 0.
    for ell, _W_ell in zip(traj_lens_unique, W_ell):
        traj_idx = history.traj_lens == ell
        # WIS
        log_rho_ell = log_rho[traj_idx]
        log_rho_ell = (log_rho_ell.T - logsumexp(log_rho_ell, axis=-1).T).T # trajectory
        rho_ell = np.exp(log_rho_ell)
        estimate_ell = np.dot(rho_ell, history.returns[traj_idx])
        # Adjust by W_ell
        estimate += _W_ell * estimate_ell # dot over trajectories
    return estimate

PHWIS.__name__ = "PHWIS_Beh"
PHWIS.__category__ = "PHIS"
PHWIS.__weighted__ = True
