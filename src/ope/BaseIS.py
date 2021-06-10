"""
Implementations of base importance sampling methods
Inputs: states (... x traj x time),
        actions (... x traj x time),
        rewards (... x traj x time),
        traj_lens (... x traj),
        pi_e,
        pi_b,
        rho_ts (... x traj x time),
        discount,
        verbose (bool)
Output: estimated discounted return (...),
        (if verbose) info as dict
If rho_ts is provided, then it will be used.
"""

# %% Imports
import numpy as np
from scipy.special import logsumexp
# %% IMPORTANCE SAMPLING METHODS

## Baseline OPE methods
def IS(history, verbose=False):
    """Importance Sampling for OPE.
    Reference: Equation 3.7 of [Thomas 2015] (p37)"""
    # work in log space
    log_rho = np.array([np.log(p).sum() for p in history.rho_ts]) # time
    log_rho -= np.log(len(history.traj_lens)) # num_trajectories
    rho = np.exp(log_rho)
    estimate = np.dot(rho, history.returns)
    if verbose:
        return estimate, {"rhos": rho, "traj_lens": history.traj_lens}
    return estimate

def WIS(history, verbose=False):
    """Weighted Importance Sampling for OPE.
    Reference: Equation 3.21 of [Thomas 2015] (p53)
    rho_ts: ... x trajectory x time
    """
    # work in log space because a very small weight may still be salvageable by the re-weighting
    log_rho = np.array([np.log(p).sum() for p in history.rho_ts]) # time
    log_rho = (log_rho.T - logsumexp(log_rho, axis=-1).T).T # trajectory
    rho = np.exp(log_rho)
    estimate = np.dot(rho, history.returns) # dot over trajectories
    if verbose:
        return estimate, {"rhos": rho, "traj_lens": history.traj_lens}
    return estimate

IS.__name__ = "IS"
IS.__category__ = "BaseIS"
IS.__weighted__ = False
WIS.__name__ = "WIS"
WIS.__category__ = "BaseIS"
WIS.__weighted__ = True
