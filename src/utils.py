"""Frequently and generally used functions"""

import os
from numbers import Integral
from itertools import combinations_with_replacement
import numpy as np
from matplotlib import colors
from tqdm import tqdm
from src.environments.gridworld.policy import EpsilonGreedyPolicy, MultiEpsilonGreedyPolicy
from src.environments.gym.policies.qfunc import EpsilonGreedyQFunctionPolicy
from src.ope.OSIRIS import make_correlation_map

class BatchOfTrajectories:
    def __init__(self, states=None, actions=None, rewards=None, traj_lens=None, actions_opt=None, states_term=None):
        self.states = states if states is not None else []                  # num_trajectories x time (staggered) x state_dim
        self.actions = actions if actions is not None else []               # num_trajectories x time (staggered)
        self.rewards = rewards if rewards is not None else []               # num_trajectories x time (staggered)
        self.traj_lens = traj_lens if traj_lens is not None else []         # num_trajectories
        self.actions_opt = actions_opt if actions_opt is not None else []   # num_trajectories x time (staggered)
        self.states_term = states_term if states_term is not None else []   # num_trajectories x state_dim

        self.returns = None # num_trajectories
        self.rho_ts = None  # num_trajectories x time (staggered)

        self.states_discrete = None # num_trajectories x time (staggered)
        self.correls_map = None # num_states
        self.correls_map_fancy = None # num_states
        self.correls_map_smirnov = None # num_states
    
    def append(self, states, actions, rewards, actions_opt, traj_len, state_term):
        self.states.append(np.array(states))
        self.actions.append(np.array(actions))
        self.rewards.append(np.array(rewards))
        self.traj_lens.append(traj_len)
        self.actions_opt.append(np.array(actions_opt))
        self.states_term.append(state_term)
    
    def reshape(self, num_trials, num_trajectories):
        assert len(self.states) >= num_trials * num_trajectories
        data = []
        for trial_num in range(num_trials):
            traj_slice = slice(trial_num * num_trajectories, (trial_num + 1) * num_trajectories)
            data.append(BatchOfTrajectories(
                states=self.states[traj_slice],
                actions=self.actions[traj_slice],
                rewards=self.rewards[traj_slice],
                traj_lens=self.traj_lens[traj_slice],
                actions_opt=self.actions_opt[traj_slice],
                states_term=self.states_term[traj_slice]
            ))
            if self.returns is not None:
                data[trial_num].returns = self.returns[traj_slice]
                data[trial_num].rho_ts = self.rho_ts[traj_slice]
            if self.states_discrete is not None or self.correls_map is not None or self.correls_map_fancy is not None or self.correls_map_smirnov is not None:
                raise NotImplementedError("need to re-generate states_discrete and correls_map and correls_map_fancy and correls_map_smirnov")
        return data

    def save_npz(self, fname):
        return np.savez_compressed(
            fname,
            states=np.concatenate(self.states, axis=-2),
            actions=np.concatenate(self.actions),
            rewards=np.concatenate(self.rewards),
            traj_lens=np.array(self.traj_lens),
            actions_opt=np.concatenate(self.actions_opt),
            states_term=np.array(self.states_term))

    def load_npz(self, fname, pbar_kwargs=None):
        loaded_file = np.load(fname)
        self.traj_lens = loaded_file["traj_lens"]
        self.states_term = loaded_file["states_term"]
        self.states = []
        self.actions = []
        self.rewards = []
        self.actions_opt = []
        self.states, self.actions, self.rewards, self.actions_opt = self._unravel(
            loaded_file["states"], loaded_file["actions"], loaded_file["rewards"], loaded_file["actions_opt"],
            pbar_kwargs=pbar_kwargs)
        return self
    
    def _unravel(self, *args, pbar_kwargs=None):
        output = tuple([] for _ in args)
        time_idx = 0
        traj_lens = self.traj_lens
        if pbar_kwargs is not None:
            traj_lens = tqdm(traj_lens, **pbar_kwargs)
        for time_max in traj_lens:
            time_slice = slice(time_idx, time_idx + time_max)
            for arg_idx, arg_in in enumerate(args):
                output[arg_idx].append(arg_in[time_slice])
            time_idx += time_max
        if len(args) == 1:
            return output[0]
        return output

    def precalculate(self, pi_e, pi_b):
        """
        Turns everything into lists of numpy arrays. Pre calculates returns and rho_ts. Assumes discount = 1
        """
        self.states = [np.array(s) for s in self.states]                    # num_trajectories x time (staggered) x state_dim
        self.actions = [np.array(a) for a in self.actions]                  # num_trajectories x time (staggered)
        self.rewards = [np.array(r) for r in self.rewards]                  # num_trajectories x time (staggered)
        self.traj_lens = np.array(self.traj_lens)                           # num_trajectories
        self.actions_opt = [np.array(a_opt) for a_opt in self.actions_opt]  # num_trajectories x time (staggered)
        self.states_term = np.array(self.states_term)                            # num_trajectories x state_dim

        self.returns = np.array([r.sum() for r in self.rewards])
        if ((isinstance(pi_e, EpsilonGreedyPolicy) and isinstance(pi_b, EpsilonGreedyPolicy)) 
                or (isinstance(pi_e, EpsilonGreedyQFunctionPolicy) and isinstance(pi_b, EpsilonGreedyQFunctionPolicy))):
            lr_subopt = pi_e._prob_subopt / pi_b._prob_subopt
            lr_opt = pi_e._prob_opt / pi_b._prob_opt
            self.rho_ts = [np.choose(a == a_opt, choices=[lr_subopt, lr_opt]) for a, a_opt in zip(self.actions, self.actions_opt)]
        elif isinstance(pi_e, EpsilonGreedyPolicy) and isinstance(pi_b, MultiEpsilonGreedyPolicy):
            lr_subopt1 = pi_e._prob_subopt / pi_b._prob_subopt1
            lr_opt1 = pi_e._prob_opt / pi_b._prob_opt1
            lr_subopt2 = pi_e._prob_subopt / pi_b._prob_subopt2
            lr_opt2 = pi_e._prob_opt / pi_b._prob_opt2
            choices = np.array([[lr_subopt1, lr_opt1], [lr_subopt2, lr_opt2]])
            self.rho_ts = [np.choose(a == abs(a_opt), choices=choices[pi_b.states_selector[s.squeeze(-1)]].T) for s, a, a_opt in zip(self.states, self.actions, self.actions_opt)]
        else:
            raise NotImplementedError()
    
    def precalculate_OSIRIS(self, states_discrete=None, correls_map=None, correls_map_fancy=None, correls_map_smirnov=None, state_discretizer=None):
        """
        Automatically unravels states_discrete as necessary.
        """
        discretized = True # outputs whether needed to calculate states_discrete
        if states_discrete is not None:
            if len(states_discrete) == len(self.traj_lens):
                self.states_discrete = states_discrete
            else:
                self.states_discrete = self._unravel(states_discrete)
        else:
            if all(s.shape[1] == 1 and issubclass(s.dtype.type, Integral) for s in self.states):
                self.states_discrete = [s.squeeze(1) for s in self.states]
                discretized = False
            else:
                states = np.concatenate(self.states)
                actions = np.concatenate(self.actions)
                states_discrete = state_discretizer(states, actions)
                self.states_discrete = self._unravel(states_discrete)
        
        if correls_map is not None:
            self.correls_map = correls_map
        else:
            states_discrete = np.concatenate(self.states_discrete)
            actions = np.concatenate(self.actions)
            rho_ts = np.concatenate(self.rho_ts)
            returns_to_go = np.concatenate([ret - np.cumsum(r) + r for ret, r in zip(self.returns, self.rewards)])
            self.correls_map = make_correlation_map(states_discrete, actions, rho_ts, returns_to_go)

        if correls_map_fancy is not None:
            self.correls_map_fancy = correls_map_fancy
        else:
            states_discrete = np.concatenate(self.states_discrete)
            actions = np.concatenate(self.actions)
            rho_ts = np.concatenate(self.rho_ts)
            returns_to_go = np.concatenate([ret - np.cumsum(r) + r for ret, r in zip(self.returns, self.rewards)])
            self.correls_map_fancy = make_correlation_map(states_discrete, actions, rho_ts, returns_to_go, fancy=True)

        if correls_map_smirnov is not None:
            self.correls_map_smirnov = correls_map_smirnov
        else:
            states_discrete = np.concatenate(self.states_discrete)
            actions = np.concatenate(self.actions)
            rho_ts = np.concatenate(self.rho_ts)
            returns_to_go = np.concatenate([ret - np.cumsum(r) + r for ret, r in zip(self.returns, self.rewards)])
            self.correls_map_smirnov = make_correlation_map(states_discrete, actions, rho_ts, returns_to_go, smirnov=True)

        return discretized

class DivergingLogNorm(colors.LogNorm):
    """
    Wrap matplotlib log norm for positive and negative values
    """
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        super(DivergingLogNorm, self).__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.vcenter = vcenter
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result, is_scalar = self.process_value(value)
        result = np.ma.masked_less_equal(result, 0, copy=False)
        self.autoscale_None(result)
        vmin, vcenter, vmax = self.vmin, self.vcenter, self.vmax
        if vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(np.ma.filled(result, fill_value=vmax), vmin, vmax),
                                     mask=mask)
            # in-place equivalent of above can be much faster
            resdat = result.data
            mask = result.mask
            if mask is np.ma.nomask:
                mask = (resdat <= 0)
            else:
                mask |= resdat <= 0
            np.copyto(resdat, 1, where=mask)
            np.log(resdat, resdat)
            result = np.ma.array(
                np.interp(resdat, [np.log(vmin), np.log(vcenter), np.log(vmax)],
                          [0, 0.5, 1.]), mask=mask, copy=False)
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)