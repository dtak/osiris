import numpy as np

def MarginalizedIS(history, verbose=False):
    if verbose:
        raise NotImplementedError()
    
    estimate = 0
    num_states = max(states_disc.max() for states_disc in history.states_discrete) + 1
    d_eva = np.empty(num_states + 1)
    r_t_est = np.zeros(num_states + 1)
    P_t = np.zeros((num_states + 1, num_states + 1))
    ONEHOT_STATES = np.eye(num_states + 1, dtype=bool)

    # Calculate d_0
    for t in range(history.traj_lens.max()): # times in trajectory
        alive_cur = t < history.traj_lens
        alive_next = t + 1 < history.traj_lens
        s_t = np.array([states_disc[t] if alive else num_states for states_disc, alive in zip(history.states_discrete, alive_cur)])
        s_t_nxt = np.array([states_disc[t + 1] if alive else num_states for states_disc, alive in zip(history.states_discrete, alive_next)])
        rho_t = np.array([p[t] if alive else 1 for p, alive in zip(history.rho_ts, alive_cur)])
        r_t = np.array([r[t] if alive else 0 for r, alive in zip(history.rewards, alive_cur)])
        if t == 0:
            d_eva, _ = np.histogram(s_t, bins=np.arange(num_states + 2))
            d_eva = d_eva / np.sum(d_eva)
        visited_cur = ONEHOT_STATES[s_t].T # num_states x num_trajectories
        visited_nxt = ONEHOT_STATES[s_t_nxt].T # num_states x num_trajectories
        visited_trans = visited_nxt[:, np.newaxis, :] * visited_cur[np.newaxis, :, :] # next_state x cur_state x num_trajectories
        n_s_t = np.sum(visited_cur, axis=1)
        r_t_est.fill(0)
        r_t_est = np.divide(np.dot(visited_cur, rho_t * r_t), n_s_t, out=r_t_est, where=n_s_t!=0) # num_states
        n_trans_t = np.sum(visited_trans, axis=2)
        P_t.fill(0)
        P_t = np.divide(np.dot(visited_trans, rho_t), n_trans_t, out=P_t, where=n_trans_t!=0) # next_state x cur_state
        estimate += np.dot(d_eva, r_t_est)
        d_eva = np.dot(P_t, d_eva)
        d_eva = d_eva / np.sum(d_eva)

    return estimate
