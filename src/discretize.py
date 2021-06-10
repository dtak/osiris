"""
Ways to discretize continuous state space for OSIRIS OPE
"""
import numpy as np

def simple_discretizer(states, actions, vals_min, vals_max, num_bins):
    """
    Receives states in continuous space, returns discretized states by binning.
    Returns ids of state; and total state count. Assume already flattened.
    """
    assert len(states.shape) == len(actions.shape) + 1
    # Make bins
    bins = [np.linspace(vmin, vmax, num=nbins) for vmin, vmax, nbins in zip(vals_min, vals_max, num_bins)]
    # Put into bins
    states_dimbin = np.array([np.digitize(s, b) for s, b in zip(states.T, bins)]).T # batch x state_dims
    # Return ids of states after they are binned and then mask up!
    _, states_discrete = np.unique(states_dimbin, axis=0, return_inverse=True)
    return states_discrete
