import numpy as np

class DiscreteMDPPolicy:
    def __init__(self, probs):
        """
        probs has shape (num_states, num_actions) which corresponds to pi(a | s)
        Discrete states, discrete actions.
        """
        assert len(probs.shape) == 2
        self.probs = probs
        self.num_states, self.num_actions = self.probs.shape

    def get_probs(self, states, actions=None):
        """
        If actions is provided, should have same shape as states, and output will have that shape.
        Otherwise, output will have extra dimension corresponding to action space.
        Can handle masked arrays.
        """
        probs = self.probs[states, actions] if actions is not None else self.probs[states]
        if np.ma.is_masked(states) or np.ma.is_masked(actions):
            assert actions is None or np.all(np.ma.getmaskarray(states) == np.ma.getmaskarray(actions))
            return np.ma.masked_array(probs, mask=np.ma.getmask(states))
        return probs

    def sample_action(self, states):
        """Samples action from each state according to policy."""
        if self.num_states is None: # continuous state space
            sample = np.random.rand(*states.shape[:-1])
        else:
            sample = np.random.rand(*states.shape)
        cum_probs = self.get_probs(states).cumsum(axis=-1) # cumulative over action space
        return (sample < cum_probs).argmax(axis=-1) # first action within interval

class DiscreteMDPEnvironment:
    def __init__(self, T, R, initial_state, terminal_states, max_time=1000):
        """
        T has shape (num_states, num_actions, num_states),
            which correspond to (s, a, s') respectively.
            Entries are transition probabilities
            p(s' | s, a).
        R will be reshaped into (num_states, num_actions,
            num_states), which correspond to (s, a, s')
            respectively. Entries are reward values for
            such a transition.
        initial_state is scalar index of initial state
        terminal_states is list of scalar indices of
            terminal states
        """
        self.num_states, self.num_actions, _ = T.shape
        self.state_space = np.arange(self.num_states)
        self.action_space = np.arange(self.num_actions)

        assert T.shape == (self.num_states, self.num_actions, self.num_states)
        non_terminal_states = np.ones(self.num_states, dtype=bool)
        non_terminal_states[terminal_states] = False
        assert np.all(T[non_terminal_states].sum(axis=2) == 1)
        assert np.all(T[non_terminal_states] >= 0)
        self.T = T

        if R.shape == (self.num_states,):
            self.R = np.tile(R[:, np.newaxis, np.newaxis], reps=(1, self.num_actions, self.num_states))
        elif R.shape == (self.num_states, self.num_actions):
            self.R = np.tile(R[:, :, np.newaxis], reps=(1, 1, self.num_states))
        elif R.shape == (self.num_states, self.num_actions, self.num_states):
            self.R = R
        else:
            raise ValueError()

        assert initial_state in self.state_space
        self.initial_state = initial_state

        assert np.all([s in self.state_space for s in terminal_states])
        self.terminal_states = np.array(terminal_states)
        # Make terminal states absorbing
        self.T[self.terminal_states, :, :] = 0.
        self.T[self.terminal_states, :, self.terminal_states] = 1.

        self.max_time = max_time

    def generate_trajectories(self, policy, num_trajectories=100, num_trials=None, pbar_kwargs=None, seed=2020, output_opt=False):
        # traj_lens doesn't include the terminal state
        np.random.seed(seed)
        if num_trials is not None:
            if output_opt:
                states, actions, rewards, traj_lens, states_ends, actions_opt = \
                    self.generate_trajectories(
                        policy=policy,
                        num_trajectories=(num_trials * num_trajectories),
                        num_trials=None,
                        pbar_kwargs=pbar_kwargs,
                        seed=seed,
                        output_opt=True
                        )
                return (
                    states.reshape((num_trials, num_trajectories,) + states.shape[1:]),
                    actions.reshape((num_trials, num_trajectories,) + actions.shape[1:]),
                    rewards.reshape((num_trials, num_trajectories,) + rewards.shape[1:]),
                    traj_lens.reshape((num_trials, num_trajectories,) + traj_lens.shape[1:]),
                    states_ends.reshape((num_trials, num_trajectories,) + states_ends.shape[1:]),
                    actions_opt.reshape((num_trials, num_trajectories,) + actions_opt.shape[1:])
                    )
            else:
                states, actions, rewards, traj_lens, states_ends = \
                    self.generate_trajectories(
                        policy=policy,
                        num_trajectories=(num_trials * num_trajectories),
                        num_trials=None,
                        pbar_kwargs=pbar_kwargs,
                        seed=seed,
                        output_opt=False
                        )
                return (
                    states.reshape((num_trials, num_trajectories,) + states.shape[1:]),
                    actions.reshape((num_trials, num_trajectories,) + actions.shape[1:]),
                    rewards.reshape((num_trials, num_trajectories,) + rewards.shape[1:]),
                    traj_lens.reshape((num_trials, num_trajectories,) + traj_lens.shape[1:]),
                    states_ends.reshape((num_trials, num_trajectories,) + states_ends.shape[1:])
                    )

        def sample_categorical(probs):
            # probs has shape batch x categories
            cum_probs = np.cumsum(probs, axis=-1)
            rnd = np.random.rand(*cum_probs.shape[:-1])
            return np.argmax(cum_probs >= rnd[..., np.newaxis], axis=-1)

        states = np.ma.masked_all((num_trajectories, self.max_time + 1), dtype=int)
        actions = np.ma.masked_all((num_trajectories, self.max_time), dtype=int)
        rewards = np.ma.masked_all((num_trajectories, self.max_time))
        actions_opt = np.ma.masked_all((num_trajectories, self.max_time), dtype=int)

        states[:, 0] = self.initial_state
        for t in range(self.max_time):
            probs_action = policy.probs[states[:, t]]
            actions[:, t] = sample_categorical(probs_action)
            actions_opt[:, t] = np.argmax(probs_action, axis=-1)
            probs_transition = self.T[states[:, t], actions[:, t]]
            states[:, t + 1] = sample_categorical(probs_transition)
            rewards[:, t] = self.R[states[:, t], actions[:, t], states[:, t + 1]]

        is_terminal = np.any(states[..., np.newaxis] == self.terminal_states, axis=-1)
        traj_lens = np.argmax(is_terminal, axis=-1)
        has_terminal = np.any(is_terminal, axis=-1) # whether each trajectory has a terminal state
        traj_lens[~has_terminal] = self.max_time

        # Mask everything after terminal
        after_terminal = np.tile(np.arange(self.max_time + 1), reps=(num_trajectories, 1)) > traj_lens[..., np.newaxis]
        states_ends = np.ma.copy(states)
        states_ends[after_terminal] = np.ma.masked
        states = states[..., :-1] # remove terminal state
        after_terminal = np.tile(np.arange(self.max_time), reps=(num_trajectories, 1)) >= traj_lens[..., np.newaxis]
        states[after_terminal] = np.ma.masked
        actions[after_terminal] = np.ma.masked
        rewards[after_terminal] = np.ma.masked
        actions_opt[after_terminal] = np.ma.masked

        if output_opt:
            return states, actions, rewards, traj_lens, states_ends, actions_opt
        else:
            return states, actions, rewards, traj_lens, states_ends

    def plot(self, decision_tree=None, data_norm=None, data_colormap='summer', custom_data=None, custom_colorbar=False, ticks=False, tight=True):
        raise NotImplementedError()

    def plot_trajectories(self, states, alpha=0.2):
        raise NotImplementedError()

    def plot_policy(self, policy):
        raise NotImplementedError()

    def plot_rhos(self, pi_e, pi_b, interpolation=None):
        raise NotImplementedError()
