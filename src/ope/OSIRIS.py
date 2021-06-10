"""
Implementation of likelihood ratio truncation by states for importance sampling
"""

import numpy as np
from scipy.stats import ttest_ind_from_stats, ks_2samp
from tqdm import trange
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.environments.gym.training.DQN import DQN

## Likelihood ratio-truncation methods

def OSIRIS(history, verbose=False, alpha=0.05, keep_map=None, mod=None):
    if verbose:
        raise NotImplementedError()
    # Only multiply together likelihood ratios for significant transitions
    if keep_map is None:
        if mod == "fancyA":
            keep_map = history.correls_map_fancy <= alpha
        elif mod == "smirnov":
            keep_map = history.correls_map_smirnov <= alpha
        else:
            assert mod is None
            keep_map = history.correls_map <= alpha
    log_rho = np.array([np.log(p[keep_map[s_disc]]).sum() for p, s_disc in zip(history.rho_ts, history.states_discrete)]) # time
    log_rho -= np.log(len(history.traj_lens)) # num_trajectories
    rho = np.exp(log_rho)
    estimate = np.dot(rho, history.returns)
    return estimate

def OSIRWIS(history, verbose=False, alpha=0.05, keep_map=None, mod=None):
    if verbose:
        raise NotImplementedError()
    # Only multiply together likelihod ratios for significant transitions
    if keep_map is None:
        if mod == "fancyA":
            keep_map = history.correls_map_fancy <= alpha
        elif mod == "smirnov":
            keep_map = history.correls_map_smirnov <= alpha
        else:
            assert mod is None
            keep_map = history.correls_map <= alpha
    log_rho = np.array([np.log(p[keep_map[s_disc]]).sum() for p, s_disc in zip(history.rho_ts, history.states_discrete)]) # time
    log_rho = (log_rho.T - logsumexp(log_rho, axis=-1).T).T # trajectory
    rho = np.exp(log_rho)
    estimate = np.dot(rho, history.returns) # dot over trajectories
    return estimate

def OSIRWIS_nn(history, verbose=False, alpha=0.4, HIDDEN_DIM=32, BATCH_SIZE=128):
    if verbose:
        raise NotImplementedError()
    # Only multiply together likelihod ratios for significant transition
    net = DQN(history.states[0].shape[-1], np.max([np.max(a) for a in history.actions]) + 1, hidden=HIDDEN_DIM)
    optimizer = optim.Adam(net.parameters())
    states = torch.tensor(np.concatenate(history.states), dtype=torch.float)
    actions = torch.tensor(np.concatenate(history.actions), dtype=torch.long).unsqueeze(1)
    returns_to_go = torch.tensor(np.concatenate([ret - np.cumsum(r) + r for ret, r in zip(history.returns, history.rewards)]), dtype=torch.float)

    for train_time in range(500):
        sample = torch.randint(states.shape[0], (BATCH_SIZE,))
        # Compute Huber loss
        loss = F.smooth_l1_loss(net(states[sample]).gather(1, actions[sample]).squeeze(1), returns_to_go[sample])
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
    diff = net(states).detach().numpy()
    diff = np.std(diff, axis=1) / np.mean(diff, axis=1)
    theta = diff > alpha
    theta = history._unravel(theta)

    log_rho = np.array([np.log(p[keep_map]).sum() for p, keep_map in zip(history.rho_ts, theta)]) # time
    log_rho_is = log_rho - np.log(len(history.traj_lens)) # num_trajectories
    rho_is = np.exp(log_rho_is)
    estimate_is = np.dot(rho_is, history.returns)
    log_rho_wis = (log_rho.T - logsumexp(log_rho, axis=-1).T).T # trajectory
    rho_wis = np.exp(log_rho_wis)
    estimate_wis = np.dot(rho_wis, history.returns) # dot over trajectories
    return estimate_is, estimate_wis

OSIRIS.__name__ = "OSIRIS"
OSIRIS.__category__ = "OSIRIS"
OSIRIS.__weighted__ = False

OSIRIS.__name__ = "OSIRWIS"
OSIRIS.__category__ = "OSIRIS"
OSIRIS.__weighted__ = True

## Truncation methods

def make_correlation_map(states_discrete, actions, rho_ts, returns_to_go, fancy=False, smirnov=False, pbar_kwargs=None):
    """
    Returns array of shape batch_dims ... x num_states where each entry is absolute
    value of covariance between trajectory return and weight in that state. Assumes
    states are already discrete (if not, user should use make_states_discrete).
    If episode_returns has a time dimension, then it will be used -- corresponding 
    to return to go.
    Assumes arguments are already flattened, representing one batch.
    """
    num_states = np.max(states_discrete) + 1
    num_actions = np.max(actions) + 1
    correls_map = np.ones(num_states)
    if pbar_kwargs is None:
        state_space = range(num_states)
    else:
        state_space = trange(num_states, **pbar_kwargs)
    for st in state_space:
        # Collect data from all visits to state
        states_idx, = np.where(states_discrete == st)
        n_st = len(states_idx)
        if n_st == 0:
            continue
        state_returns = returns_to_go[states_idx]
        # Binarize actions
        if fancy:
            state_actions = actions[states_idx]
            ret_mid = np.median(state_returns)
            ret_pos = np.random.rand(n_st) < 0.5 # random tie breaking
            ret_pos[state_returns > ret_mid] = True
            ret_pos[state_returns < ret_mid] = False
            actions_onehot = np.eye(num_actions, dtype=int)[state_actions].T # num_actions x visits
            actions_pos_freq = np.dot(actions_onehot, ret_pos) / np.sum(actions_onehot, axis=1) # num_actions
            actions_class = np.random.randn(num_actions) < 0.5 # random tie breaking
            actions_class[actions_pos_freq > 0.5] = True
            actions_class[actions_pos_freq < 0.5] = False
            lr_pos = actions_class[state_actions]
        else:
            lr_pos = rho_ts[states_idx] > 1
        lr_neg = ~lr_pos
        state_returns_pos = state_returns[lr_pos]
        state_returns_neg = state_returns[lr_neg]
        # t-test
        if len(state_returns_pos) > 0 and len(state_returns_neg) > 0:
            if not smirnov:
                correls_map[st] = ttest_ind_from_stats(
                    state_returns_pos.mean(), state_returns_pos.std(), lr_pos.sum(),
                    state_returns_neg.mean(), state_returns_neg.std(), lr_neg.sum(),
                    equal_var=False)[1]
            else:
                correls_map[st] = ks_2samp(
                    state_returns_pos,
                    state_returns_neg)[1]
        else:
            continue
    return correls_map
