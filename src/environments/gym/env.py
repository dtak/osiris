"""Wraps OpenAI gym"""
import numpy as np
import gym
from tqdm import trange
from src.environments.mdp import DiscreteMDPEnvironment
from src.utils import BatchOfTrajectories

class GymEnvironment(DiscreteMDPEnvironment):
    def __init__(self, name, trajectories=None, state_dim_names=None, action_names=None, unwrapped=False):
        """
        `name` is of OpenAI gym. `trajectories` is dictionary of
        (policy object) => filename format of pre-generated trajectories.
        Assume state space is continuous and action space is discrete.
        """
        self.env = gym.make(name)
        if unwrapped:
            self.env = self.env.unwrapped
        self.trajectories = trajectories

        self.state_dim = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.n

        self.state_dim_names = state_dim_names
        if self.state_dim_names is None:
            self.state_dim_names = ["state[{:d}]".format(i) for i in range(self.state_dim)]
        assert len(self.state_dim_names) == self.state_dim

        self.action_names = action_names
        if self.action_names is None:
            self.action_names = []
        assert len(self.action_names) == self.action_num

    def generate_trajectories(self, policy, num_trajectories=100, num_trials=None, pbar_kwargs=None, seed=2020, render=False, output_opt=False):
        total_num_trajectories = num_trajectories if num_trials is None else num_trajectories * num_trials

        history = BatchOfTrajectories()
        for traj_num in trange(total_num_trajectories, **pbar_kwargs):
            np.random.seed(seed + traj_num)
            trajectory = []
            s = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                # Pick action
                a, a_opt = policy.sample_action(s, output_opt=True)
                # Get response from environment
                s_next, r, done, _ = self.env.step(a)
                trajectory.append((s, a, r, a_opt))
                # Transition to next state
                s = s_next
            history.append(*zip(*trajectory), traj_len=len(trajectory), state_term=s)
        self.env.close()

        if num_trials is not None:
            return history.reshape(num_trials, num_trajectories)
        return history
