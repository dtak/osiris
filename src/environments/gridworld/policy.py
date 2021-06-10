import numpy as np
from src.environments.mdp import DiscreteMDPPolicy

class EpsilonGreedyPolicy(DiscreteMDPPolicy):
    def __init__(self, policy, epsilon, directions="NSEW"):
        """
        Input: n x m matrix of direction to be taken from each state in the maze
        Output: n x m x a matrix of probabilities to taking each action from each state
        """
        random_prob = epsilon / len(directions)
        action_probs = (np.eye(len(directions)) * (1 - epsilon)) + random_prob
        probs = np.array([action_probs[directions.index(action_dir)] for action_dir in "".join(policy)])
        self.actions_opt = np.array([directions.index(action_dir) for action_dir in "".join(policy)])
        self._prob_subopt = epsilon / len(directions)
        self._prob_opt = 1. - epsilon + self._prob_subopt
        super().__init__(probs=probs)

class MultiEpsilonGreedyPolicy(DiscreteMDPPolicy):
    def __init__(self, policy, epsilon1, directions1="NSEW", epsilon2=None, directions2="nsew"):
        """
        Input: n x m matrix of direction to be taken from each state in the maze
        Output: n x m x a matrix of probabilities to taking each action from each state
        """
        action_probs1 = (np.eye(len(directions1)) * (1 - epsilon1)) + (epsilon1 / len(directions1))
        action_probs2 = (np.eye(len(directions2)) * (1 - epsilon2)) + (epsilon2 / len(directions2))
        probs = np.array([action_probs1[directions1.index(action_dir)] if action_dir in directions1 else action_probs2[directions2.index(action_dir)] for action_dir in "".join(policy)])
        self.states_selector = np.array([action_dir in directions2 for action_dir in "".join(policy)], dtype=int)
        self.actions_opt = np.array([directions1.index(action_dir) if action_dir in directions1 else directions2.index(action_dir) for action_dir in "".join(policy)])
        self._prob_subopt1 = epsilon1 / len(directions1)
        self._prob_opt1 = 1. - epsilon1 + self._prob_subopt1
        self._prob_subopt2 = epsilon2 / len(directions2)
        self._prob_opt2 = 1. - epsilon2 + self._prob_subopt2
        super().__init__(probs=probs)
