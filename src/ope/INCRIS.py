import numpy as np

def INCRIS(history, verbose=False):
    if verbose:
        raise NotImplementedError()
    estimate = 0
    A_k_all = [np.insert(p.cumprod(), 0, values=1.) for p in history.rho_ts] # num_trajectories x (0, 1, ..., traj_len)
    B_k_all = [_A_k[-1] / _A_k for _A_k in A_k_all] # num_trajectories x (0, 1, ..., traj_len)
    for t in range(1, history.traj_lens.max() + 1): # times in trajectory
        r_t = np.array([r[t - 1] for r, traj_len in zip(history.rewards, history.traj_lens) if t <= traj_len]) # num_trajectories
        k_vals = np.arange(0, t + 1)
        A_k = np.array([_A_k[t - k_vals] for _A_k, traj_len in zip(A_k_all, history.traj_lens) if t <= traj_len]) # num_trajectories x k_vals
        B_k = np.array([_B_k[t - k_vals] for _B_k, traj_len in zip(B_k_all, history.traj_lens) if t <= traj_len]) # num_trajectories x k_vals
        Bkrt = B_k * r_t[:, np.newaxis] # num_trajectories x k_vals
        Bkrt_dev = Bkrt - np.mean(Bkrt, axis=0) # mean over trajectories in batch; num_trajectories x k_vals
        C_k_hat = np.mean((A_k - np.mean(A_k, axis=0)) * (Bkrt_dev), axis=0) # k_vals
        V_k_hat = np.mean(Bkrt_dev ** 2, axis=0) # k_vals
        MSE_k_hat = C_k_hat ** 2 + V_k_hat # k_vals
        k_prime = np.argmin(MSE_k_hat) # scalar
        Bkrt_prime = Bkrt[:, k_prime] # num_trajectories
        r_t_hat = np.mean(Bkrt_prime) # scalar
        estimate += r_t_hat
    return estimate
