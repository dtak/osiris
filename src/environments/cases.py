"""
GridworldDD: Dilly-Dallying Gridworld.
GridworldXP: Express Gridworld
CartPole: OpenAI gym Cart Pole.
LunarLander: OpenAI gym Lunar Lander.
"""

import numpy as np
import torch
from src.environments.gridworld.gridworld import GridWorld
from src.environments.gridworld.policy import EpsilonGreedyPolicy, MultiEpsilonGreedyPolicy
from src.environments.gym.training.DQN import DQN
from src.environments.gym.policies.qfunc import EpsilonGreedyQFunctionPolicy
from src.environments.gym.env import GymEnvironment
from src.discretize import simple_discretizer

def make_env(identifier):
    """
    Storage of noteworthy environments and policies
    Returns: env, pi_e, pi_b, num_trajectories, discount, state_discretizer.
    """
    if identifier in ("GridworldDD", "GridworldXP"):
        num_trajectories = 25
        discount = 1.
        env = GridWorld([
            "##############",
            "#,,,,,,,,..Z.#",
            "#,#######....#",
            "#,#######Z...#",
            "#o..........A#",
            "#ZZZZZZZZZZZZ#",
            "##############",
        ], rewards={'A': +5, 'Z': -5, 'moved': 0.}, terminal_markers='AZ', max_time=100)
        pi_e = EpsilonGreedyPolicy([
            "NNNNNNNNNNNNNN",
            "NEEEEEEEEESNSN",
            "NNNNNNNNNEESSN",
            "NNNNNNNNNEEESN",
            "NNWEEEEEEENENN",
            "NNNNNNNNNNNNNN",
            "NNNNNNNNNNNNNN",
        ], epsilon=0.1)
        if identifier == "GridworldXP":
            pi_b = MultiEpsilonGreedyPolicy([
                "NNNNNNNNNNNNNN",
                "NeeeeeeeeESNSN",
                "NnNNNNNNNEESSN",
                "NnNNNNNNNEEESN",
                "NNWEEEEEEENENN",
                "NNNNNNNNNNNNNN",
                "NNNNNNNNNNNNNN",
            ], epsilon1=0.5, epsilon2=0.2)
        else:
            pi_b = EpsilonGreedyPolicy([
                "NNNNNNNNNNNNNN",
                "NEEEEEEEEESNSN",
                "NNNNNNNNNEESSN",
                "NNNNNNNNNEEESN",
                "NNWEEEEEEENENN",
                "NNNNNNNNNNNNNN",
                "NNNNNNNNNNNNNN",
            ], epsilon=0.5)
        state_discretizer = None
    elif identifier == "CartPole":
        env = GymEnvironment(
            "CartPole-v1",
            state_dim_names=("Position", "Velocity", "Angle", "Angular Velocity"),
            action_names=("Push left", "Push right"),
            unwrapped=True
            )
        Q = DQN(env.state_dim, env.action_num)
        Q.load_state_dict(torch.load("./data/CartPole-v1/Qfunc_optimal_20200818.torch"))
        pi_e = EpsilonGreedyQFunctionPolicy(Q, epsilon=0.20, num_actions=env.action_num)
        pi_b = EpsilonGreedyQFunctionPolicy(Q, epsilon=0.25, num_actions=env.action_num)
        num_trajectories = 50
        discount = 1.
        def state_discretizer(states, actions):
            return simple_discretizer(states, actions, vals_min=np.array([-1, -3, -1, -3]), vals_max=np.array([1, 3, 1, 3]), num_bins=np.array([3, 3, 3, 3]))
    elif identifier == "LunarLander":
        env = GymEnvironment(
            "LunarLander-v2",
            state_dim_names=("x position", "y position", "x velocity", "y velocity", "angle", "angular velocity", "leg 1 contact", "leg 2 contact"),
            action_names=("NOP", "Left engine", "Main engine", "Right engine")
            )
        Q = DQN(env.state_dim, env.action_num, hidden=64)
        Q.load_state_dict(torch.load("./data/LunarLander-v2/Qfunc_optimal.torch"))
        pi_e = EpsilonGreedyQFunctionPolicy(Q, epsilon=0.05, num_actions=env.action_num)
        pi_b = EpsilonGreedyQFunctionPolicy(Q, epsilon=0.1, num_actions=env.action_num)
        num_trajectories = 50
        discount = 1.
        def state_discretizer(states, actions):
            return simple_discretizer(states, actions, vals_min=np.array([-1, -1, -1, -1, -1, -1, 0, 0]), vals_max=np.array([1, 1, 1, 1, 1, 1, 1, 1]), num_bins=np.array([3, 3, 3, 3, 3, 3, 2, 2]))
    else:
        raise ValueError()
    return env, pi_e, pi_b, num_trajectories, discount, state_discretizer
