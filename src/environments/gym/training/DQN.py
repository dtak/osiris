"""
DQN model, ReplayMemory queue, and Transition named tuple. As script, trains DQN on OpenAI gym.
Mostly adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import math
import random
from collections import namedtuple
from itertools import count
import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs, hidden=24):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(inputs, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin4 = nn.Linear(hidden, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin4(x)
        return x

# Trains a DQN
if __name__ == "__main__":
    # CartPole-v1: use EPS_END=0.01, EPS_DECAY=200 steps, TARGET_UPDATE=10 episodes, RETURN_SUCCESS=195
    # LunarLander-v2: user EPS_END=0.05, EPS_DECAY=100 episodes, TARGET_UPDATE=1000 steps, RETURN_SUCCESS=240
    CHECKPOINT_FNAME = "./"
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 100
    TARGET_UPDATE = 1000
    ENVIRONMENT_NAME = "LunarLander-v2"
    RETURN_SUCCESS = 240
    kernel_size = 3 # smoothing filter for plots
    CHECKPOINT_UPDATE = 100 #episodes
    RENDER = False

    MEMORY_SIZE = 10000
    HIDDEN_DIM = 64

    env = gym.make(ENVIRONMENT_NAME)

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(env.observation_space.shape[0], n_actions, hidden=HIDDEN_DIM)
    target_net = DQN(env.observation_space.shape[0], n_actions, hidden=HIDDEN_DIM)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters()) # default lr at 1e-3 the =n 5e-4 at episode 200
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=[200, 400])
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0 # used for epsilon decay
    episode_returns = []
    episode_returns_smooth = []
    epsilon_values = []
    loss_values = []
    q_values = []

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss.item()

    # Keep training until win condition, defined by OpenAI
    i_episode = 0
    pbar_episode = tqdm(desc="Episode", position=0)
    while len(episode_returns) < 100 or episode_returns_smooth[i_episode - 1] < RETURN_SUCCESS:
        # Initialize the environment and state
        episode_return = 0
        state = torch.tensor([env.reset()], dtype=torch.float)
        episode_loss = []
        episode_q = []
        for t in tqdm(count(), desc="Step", position=1, leave=False):
            if RENDER:
                env.render()
            # Select action
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * i_episode / EPS_DECAY)
            epsilon_values.append(eps_threshold)
            steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    q_val, action = policy_net(state).max(1)
                    episode_q.append(q_val)
                    action = action.view(1, 1)
            else:
                action = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
            # Perform action and observe new state
            next_state, reward, done, _ = env.step(action.item())
            episode_return += reward
            reward = torch.tensor([reward], dtype=torch.float)
            if done:
                next_state = None
            else:
                next_state = torch.tensor([next_state], dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model()
            if loss:
                episode_loss.append(loss)
            if done:
                episode_returns.append(episode_return)
                episode_returns_smooth.append(np.mean(episode_returns[-100:]))
                lr_scheduler.step()
                loss_values.append(np.mean(episode_loss) if len(episode_loss) > 0 else float("nan"))
                q_values.append(np.mean(episode_q))
                break
            # Update the target network, copying all weights and biases in DQN
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        # Occasionally save a checkpoint of training state
        if i_episode % CHECKPOINT_UPDATE == 0:
            torch.save(target_net.state_dict(), CHECKPOINT_FNAME + "target_net_checkpoint{}.torch".format(i_episode))
            torch.save(policy_net.state_dict(), CHECKPOINT_FNAME + "Qfunc_checkpoint{}.torch".format(i_episode))
            torch.save(optimizer.state_dict(), CHECKPOINT_FNAME + "optimizer_checkpoint{}.torch".format(i_episode))
            torch.save(lr_scheduler.state_dict(), CHECKPOINT_FNAME + "lr_scheduler_checkpoint{}.torch".format(i_episode))
            pickle.dump(memory, open(CHECKPOINT_FNAME + "memory_checkpoint{}.pkl".format(i_episode), "wb"))

        # And update plots too
        plt.figure(1)
        plt.cla()
        plt.plot(episode_returns)
        plt.plot(episode_returns_smooth)
        plt.gca().set_ylim(top=300, bottom=-300)
        plt.axhline(RETURN_SUCCESS, color="k", linestyle="--")
        plt.pause(0.01)
        plt.figure(2)
        plt.cla()
        plt.plot(scipy.signal.medfilt(loss_values, kernel_size=kernel_size) if len(loss_values) > kernel_size else loss_values)
        plt.pause(0.01)
        plt.figure(3)
        plt.cla()
        plt.plot(scipy.signal.medfilt(q_values, kernel_size=kernel_size) if len(q_values) > kernel_size else q_values)
        plt.pause(0.01)
        i_episode += 1
        pbar_episode.update()

    # After training, save policy_net which ran the successful trajectories
    torch.save(policy_net.state_dict(), CHECKPOINT_FNAME + "Qfunc_optimal.torch")
    np.save(CHECKPOINT_FNAME + "episode_returns.npy", episode_returns)
    np.save(CHECKPOINT_FNAME + "loss_values.npy", loss_values)
    np.save(CHECKPOINT_FNAME + "epsilon_values.npy", epsilon_values)
    pbar_episode.close()
    env.close()
