"""
Wrappers for Q-functions as epsilon-greedy policies
"""
import torch
import numpy as np
from src.environments.mdp import DiscreteMDPPolicy

class EpsilonGreedyQFunctionPolicy(DiscreteMDPPolicy):
    def __init__(self, Q, epsilon, num_actions):
        """
        Optimal policy but take random action
        with probability `epsilon`.
        Continuous states, discrete actions.
        - `Q` is a pytorch nn model.
        """
        self.probs = None
        self.Q = Q
        self.Q.eval()
        self.epsilon = epsilon
        self.num_states = None
        self.num_actions = num_actions
        # Precalculate prob of taking (sub)optimal actions
        self._prob_subopt = self.epsilon / self.num_actions
        self._prob_opt = 1. - self.epsilon + self._prob_subopt

    def get_probs(self, states, actions=None, output_opt=False):
        """States should have state_dim. Can handle masked arrays."""
        # Get only unmasked values
        masked_not = ~np.ma.getmaskarray(states).any(axis=-1)
        assert actions is None or np.all(masked_not == ~np.ma.getmaskarray(actions))
        states_flat = states[masked_not]
        actions_flat = actions[masked_not] if actions is not None else None
        # Calculate Q function and optimal actions
        Q_values = self._evaluate_qval(states_flat)
        actions_opt = Q_values.argmax(axis=-1)
        # Consider suboptimal actions by eps-greedy
        probs = np.full_like(Q_values, fill_value=self._prob_subopt)
        np.put_along_axis(probs, actions_opt[..., np.newaxis], values=self._prob_opt, axis=-1)
        # Make proper shape as necessary
        if actions_flat is not None:
            actions_unmask = np.ma.filled(actions_flat[..., np.newaxis], fill_value=0)
            probs = np.take_along_axis(probs, actions_unmask, axis=-1).squeeze(-1)
            output = np.ma.masked_all(states.shape[:-1])
            if output_opt:
                raise NotImplementedError()
        else:
            output = np.ma.masked_all((*states.shape[:-1], self.num_actions))
            if output_opt:
                output_actions_opt = np.ma.masked_all(states.shape[:-1])
                output_actions_opt[masked_not] = actions_opt
        output[masked_not] = probs
        # Re-mask as necessary
        if np.ma.is_masked(states) or np.ma.is_masked(actions):
            if output_opt:
                return output, output_actions_opt
            return output
        else:
            assert not np.ma.is_masked(output)
            if output_opt:
                assert not np.ma.is_masked(output_actions_opt)
                return np.ma.getdata(output), np.ma.getdata(output_actions_opt)
            return np.ma.getdata(output)

    def sample_action(self, states, output_opt=False):
        """Samples action from each state according to policy."""
        if self.num_states is None: # continuous state space
            sample = np.random.rand(*states.shape[:-1])
        else:
            sample = np.random.rand(*states.shape)
        probs = self.get_probs(states, output_opt=output_opt)
        if output_opt:
            probs, actions_opt = probs
        cum_probs = probs.cumsum(axis=-1) # cumulative over action space
        actions = (sample < cum_probs).argmax(axis=-1) # first action within interval
        if output_opt:
            return actions, actions_opt
        return actions

    def _evaluate_qval(self, states):
        """Returns Q function estimate as numpy."""
        with torch.no_grad():
            return self.Q(torch.tensor(states, dtype=torch.float)).detach().numpy()
