import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import numpy as np
from tqdm import trange
from scipy.stats import norm
from src.utils import DivergingLogNorm, BatchOfTrajectories
from src.environments.mdp import DiscreteMDPEnvironment, DiscreteMDPPolicy

# This gridworld is adapted from [redacted for double-blind review]

# Maze state is represented as a 2-element NumPy array: (Y, X). Increasing Y is South.

# Possible actions, expressed as (delta-y, delta-x).
maze_actions = {
    'N': np.array([-1, 0]),
    'S': np.array([1, 0]),
    'E': np.array([0, 1]),
    'W': np.array([0, -1]),
}

def parse_topology(topology):
    return np.array([list(row) for row in topology])

class Maze(object):
    """
    Simple wrapper around a NumPy 2D array to handle flattened indexing and staying in bounds.
    """
    def __init__(self, topology):
        self.topology = parse_topology(topology)
        self.flat_topology = self.topology.ravel()
        self.shape = self.topology.shape

    def in_bounds_flat(self, position):
        return 0 <= position < np.product(self.shape)

    def in_bounds_unflat(self, position):
        return 0 <= position[0] < self.shape[0] and 0 <= position[1] < self.shape[1]

    def get_flat(self, position):
        if not self.in_bounds_flat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.flat_topology[position]

    def get_unflat(self, position):
        if not self.in_bounds_unflat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.topology[tuple(position)]

    def flatten_index(self, index_tuple):
        return np.ravel_multi_index(index_tuple, self.shape)

    def unflatten_index(self, flattened_index):
        return np.unravel_index(flattened_index, self.shape)

    def flat_positions_containing(self, x):
        return list(np.nonzero(self.flat_topology == x)[0])

    def flat_positions_not_containing(self, x):
        return list(np.nonzero(self.flat_topology != x)[0])
    
    @property
    def start_coords(self):
        return self.unflatten_index(self.flat_positions_containing('o')[0])

    @property
    def goal_coords(self):
        goals = self.flat_positions_containing('*')
        if len(goals) == 0:
            return None
        return self.unflatten_index(goals[0])

    def __str__(self):
        return '\n'.join(''.join(row) for row in self.topology.tolist())

    def __repr__(self):
        return 'Maze({})'.format(repr(self.topology.tolist()))

def move_avoiding_walls(maze, position, action):
    """
    Return the new position after moving, and the event that happened ('hit-wall' or 'moved').

    Works with the position and action as a (row, column) array.
    """
    # Compute new position
    new_position = position + action

    # Compute collisions with walls, including implicit walls at the ends of the world.
    if not maze.in_bounds_unflat(new_position) or maze.get_unflat(new_position) == '#':
        return position, 'hit-wall'

    return new_position, 'moved'

class GridWorld(DiscreteMDPEnvironment):
    """
    A simple task in a maze: get to the goal.

    Parameters
    ----------

    maze : list of strings or lists
        maze topology (see below)

    rewards: dict of string to number. default: {'*': 10}.
        Rewards obtained by being in a maze grid with the specified contents,
        or experiencing the specified event (either 'hit-wall' or 'moved'). The
        contributions of content reward and event reward are summed. For
        example, you might specify a cost for moving by passing
        rewards={'*': 10, 'moved': -1}.

    terminal_markers: sequence of chars, default '*'
        A grid cell containing any of these markers will be considered a
        "terminal" state.

    action_error_prob: float
        With this probability, the requested action is ignored and a random
        action is chosen instead.

    Notes
    -----

    Maze topology is expressed textually. Key:
     '#': wall
     '.': open (really, anything that's not '#')
     '*': goal
     'o': origin
     'X': pitfall
    """

    def __init__(self, maze, rewards={'*': 10}, terminal_markers='*', action_error_prob=0, directions="NSEW", max_time=1000):
        self.directions = directions

        self.maze_dimensions = (len(maze), len(maze[0]))

        self.maze = Maze(maze) if not isinstance(maze, Maze) else maze
        self.rewards = rewards
        self.terminal_markers = terminal_markers
        self.action_error_prob = action_error_prob

        self.actions = [maze_actions[direction] for direction in directions]
        self.num_actions = len(self.actions)
        self.binary_action_codes = np.arange(self.num_actions).reshape(-1,1)
        self.state = None
        self.reset()
        self.num_states = self.maze.shape[0] * self.maze.shape[1]
        
        super().__init__(*self.as_mdp(), max_time=max_time)
        
    def __repr__(self):
        return 'GridWorld(maze={maze!r}, rewards={rewards}, terminal_markers={terminal_markers}, action_error_prob={action_error_prob})'.format(**self.__dict__)

    def reset(self):
        """
        Reset the position to a starting position (an 'o'), chosen at random.
        """
        options = self.maze.flat_positions_containing('o')
        self.state = options[ np.random.choice(len(options)) ]

    def is_terminal(self, state):
        """Check if the given state is a terminal state."""
        return self.maze.get_flat(state) in self.terminal_markers

    def is_done(self):
        return self.is_terminal(self.state)

    def is_cliff(self, state):
        """Check if the given state is a cliff state."""
        return self.maze.get_flat(state) == 'X'

    def observe(self):
        """
        Return the current state as an integer.

        The state is the index into the flattened maze.
        """
        return np.array([self.state])

    def perform(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        # In the absorbing end state, nothing does anything.
        if self.is_terminal(self.state):
            return 0, self.observe()

        if self.action_error_prob and np.random.rand() < self.action_error_prob:
            if np.random.rand() < .5:
                if action_idx == 0 or action_idx == 1:
                    action_idx = 2
                else: 
                    action_idx = 0
            else: 
                if action_idx == 0 or action_idx == 1:
                    action_idx = 3
                else: 
                    action_idx = 1
            
        action = self.actions[action_idx]
        new_state_tuple, result = move_avoiding_walls(self.maze, self.maze.unflatten_index(self.state), action)
        self.state = self.maze.flatten_index(new_state_tuple)

        reward = self.rewards.get(self.maze.get_flat(self.state), 0) + self.rewards.get(result, 0)
        return reward, self.observe()

    def as_mdp(self):
        transition_probabilities = np.zeros((self.num_states, self.num_actions, self.num_states))
        rewards = np.zeros((self.num_states, self.num_actions, self.num_states))
        action_rewards = np.zeros((self.num_states, self.num_actions))
        destination_rewards = np.zeros(self.num_states)

        for state in range(self.num_states):
            destination_rewards[state] = self.rewards.get(self.maze.get_flat(state), 0)

        is_terminal_state = np.zeros(self.num_states, dtype=np.bool)

        for state in range(self.num_states):
            if self.is_terminal(state):
                is_terminal_state[state] = True
                transition_probabilities[state, :, state] = 1.
            else:
                for action in range(self.num_actions):
                    new_state_tuple, result = move_avoiding_walls(self.maze, self.maze.unflatten_index(state), self.actions[action])
                    new_state = self.maze.flatten_index(new_state_tuple)
                    transition_probabilities[state, action, new_state] = 1.
                    action_rewards[state, action] = self.rewards.get(result, 0)

        # Now account for action noise.
        transitions_given_random_action = transition_probabilities.mean(axis=1, keepdims=True)
        transition_probabilities *= (1 - self.action_error_prob)
        transition_probabilities += self.action_error_prob * transitions_given_random_action

        rewards_given_random_action = action_rewards.mean(axis=1, keepdims=True)
        action_rewards = (1 - self.action_error_prob) * action_rewards + self.action_error_prob * rewards_given_random_action
        rewards = action_rewards[:, :, None] + destination_rewards[None, None, :]
        rewards[is_terminal_state] = 0

        initial_state = self.maze.flat_positions_containing('o')
        if not len(initial_state) == 1:
            raise NotImplementedError()

        return transition_probabilities, rewards, initial_state[0], np.nonzero(is_terminal_state)[0]

    def get_max_reward(self):
        _, rewards, _, _ = self.as_mdp()
        return rewards.max()
    
    def plot(self, decision_tree=None, data_norm=None, data_colormap='summer', custom_data=None, custom_colorbar=False, ticks=False, tight=True, fontsize_startend=14):
        """Visualize a policy on the maze."""
        row_count, col_count = self.maze_dimensions
        maze_dims = (row_count, col_count)
        if custom_data is not None:
            if not (len(custom_data.shape) == len(self.maze_dimensions) and np.all(custom_data.shape == self.maze_dimensions)):
                custom_data = custom_data.reshape(self.maze_dimensions)
            rewards = custom_data
            custom_colorbar = True
        else:
            rewards = np.zeros(maze_dims)
        wall_info = .5 + np.zeros(maze_dims)
        wall_mask = np.zeros(maze_dims)
        for row in range(row_count):
            for col in range(col_count):
                if self.maze.topology[row][col] == '#':
                    wall_mask[row,col] = 1
                if custom_data is None:
                    rewards[row,col] = self.rewards.get(self.maze.topology[row][col], 0) + self.rewards.get("moved", 0) # assume successfully moved
        wall_info = np.ma.masked_where(wall_mask==0, wall_info)
        rewards *= (1-wall_mask)**2
        rewards_plot = plt.pcolormesh(np.arange(-0.5, col_count), np.arange(-0.5, row_count), rewards, norm=data_norm, cmap=data_colormap)
        plt.pcolormesh(np.arange(-0.5, col_count), np.arange(-0.5, row_count), wall_info, cmap='gray')
        y,x = self.maze.start_coords
        plt.text(x,y,'start', color='gray', fontsize=fontsize_startend, va='center', ha='center', fontweight='bold')
        for row in range(row_count):
            for col in range(col_count):
                if self.maze.topology[row][col] in self.terminal_markers:
                    y,x = row,col
                    plt.text(x,y,'end', color='gray', fontsize=fontsize_startend, va='center', ha='center', fontweight='bold')
        # Show only half of the border wall states
        if tight:
            nonwall_x = np.nonzero(np.any(wall_mask == 0, axis=0))[0]
            nonwall_y = np.nonzero(np.any(wall_mask == 0, axis=1))[0]
            plt.xlim(left=nonwall_x[0] - 1, right=nonwall_x[-1] + 1)
            plt.ylim(top=nonwall_y[0] - 1, bottom=nonwall_y[-1] + 1)
        # Show or hide ticks
        if ticks:
            plt.xlabel("X-Coordinate")
            plt.ylabel("Y-Coordinate")
        else:
            plt.xticks([])
            plt.yticks([])
        # Show or hide reward function colorbar
        if custom_colorbar:
            return rewards_plot
        else:
            plt.colorbar(rewards_plot, label='Reward Function', orientation='horizontal')

    def plot_trajectories(self, states, alpha=0.2, time_colored=True):
        """
        Visualize trajectories on maze.
        If `time_colored`, then points on trajectory lines are colored by
            visitation time.
        """
        assert len(states.shape) == 2 # num_trajectories, time
        if time_colored:
            assert isinstance(alpha, float)
        else:
            if isinstance(alpha, float):
                alpha = alpha * np.ones(len(states))
            elif isinstance(alpha, np.ndarray):
                assert(len(alpha) == len(states))
            else:
                raise AssertionError()
        alpha = np.clip(alpha, 1e-2, 1.)

        states_yx = np.ma.masked_array( # num_trajectories, time, (y, x)
            np.stack(self.maze.unflatten_index(states.filled(0)), axis=2),
            mask=np.repeat(np.ma.getmaskarray(states)[..., np.newaxis], repeats=2, axis=2)
            )
        states_yx = states_yx + (0.15 * np.random.randn(*states_yx.shape)).clip(-0.5, 0.5) # add randomness so that state visitation density is clearer
        if time_colored:
            segments_mask = np.ma.getmaskarray(states)[:, 1:] # num_trajectories, time-1 whether destination is masked
            segments_mask = segments_mask.reshape(-1)

            time_idx = np.arange(states.shape[1] - 1)
            time_idx = np.tile(time_idx, reps=(states.shape[0], 1)) # num_trajectories, time
            time_idx = time_idx.reshape(-1)
            time_idx = np.array([val for val, msk in zip(time_idx, segments_mask) if not msk])

            points = np.flip(states_yx, axis=2) # num_trajectories, time, (x, y)
            segments = np.stack([points[:, :-1], points[:, 1:]], axis=2)
            segments = segments.reshape(-1, 2, 2)
            segments = np.array([val for val, msk in zip(segments, segments_mask) if not msk])

            time_norm = plt.Normalize(time_idx.min(), time_idx.max())
            lc = LineCollection(segments, cmap='autumn', norm=time_norm, alpha=alpha)
            lc.set_array(time_idx) # Set the values used for colormapping
            line = plt.gca().add_collection(lc)
            cb = plt.colorbar(line, ax=plt.gca(), label="Time")
            cb.solids.set(alpha=1.)
        else:
            for s, a in zip(states_yx, alpha):
                plt.plot(s[:, 1], s[:, 0], color="red", alpha=a)

    def plot_policy(self, policy):
        """Visualize a policy on the maze."""
        row_count, col_count = self.maze_dimensions
        maze_dims = (row_count, col_count)
        wall_info = .5 + np.zeros(maze_dims)
        wall_mask = np.zeros(maze_dims)
        for row in range(row_count):
            for col in range(col_count):
                if self.maze.topology[row][col] == '#':
                    wall_mask[row,col] = 1
        wall_info = np.ma.masked_where(wall_mask==0, wall_info)
        for row in range( row_count ):
            for col in range( col_count ):
                if wall_mask[row][col] == 1 or self.maze.get_unflat((row, col)) in self.terminal_markers:
                    continue
                probs_from_state = policy.get_probs(self.maze.flatten_index((row, col)))
                for a, prob in enumerate(probs_from_state):
                    if prob > 0:
                        dy, dx = 0.5 * self.actions[a] * prob
                        alpha = 0.2 + 0.6 * prob / probs_from_state.max() # normalize to [0.2, 0.8] so that everything is still visible
                        plt.arrow(col, row, dx, dy,
                            shape='full', facecolor='r', edgecolor='r', linewidth=0.5, length_includes_head=False, head_width=.1, alpha=alpha)

    def plot_rhos(self, pi_e, pi_b, interpolation=None):
        row_count, col_count = self.maze_dimensions
        maze_dims = (row_count, col_count)
        wall_info = .5 + np.zeros(maze_dims)
        wall_mask = np.zeros(maze_dims)
        for row in range(row_count):
            for col in range(col_count):
                if self.maze.topology[row][col] == '#':
                    wall_mask[row,col] = 1
        wall_info = np.ma.masked_where(wall_mask==0, wall_info)
        ax = plt.gca()

        lrs = np.ma.masked_equal(np.ma.masked_invalid(pi_e.probs / pi_b.probs), 0) # num_states x num_actions
        def draw_triangles(x, y, c, meet=3/16, end=7/16):
            # c is NSEW
            top_l = [x - meet, y - meet]
            top_r = [x + meet, y - meet]
            bot_l = [x - meet, y + meet]
            bot_r = [x + meet, y + meet]
            top_c = [x, y - end]
            mid_r = [x + end, y]
            bot_c = [x, y + end]
            mid_l = [x - end, y]
            mid_c = [x, y]
            ax.add_patch(plt.Polygon([mid_c, top_l, top_c, top_r], facecolor=c[0])) # north
            ax.add_patch(plt.Polygon([mid_c, top_r, mid_r, bot_r], facecolor=c[2])) # east
            ax.add_patch(plt.Polygon([mid_c, bot_r, bot_c, bot_l], facecolor=c[1])) # south
            ax.add_patch(plt.Polygon([mid_c, bot_l, mid_l, top_l], facecolor=c[3])) # west
        cmap = cm.ScalarMappable(cmap="bwr", norm=DivergingLogNorm(vmin=lrs.min(), vcenter=1., vmax=lrs.max()))
        for state_idx, state_lrs in enumerate(lrs):
            y, x = self.maze.unflatten_index(state_idx)
            if not wall_info[y, x] and not self.maze.topology[y][x] in self.terminal_markers:
                draw_triangles(x, y, [cmap.to_rgba(lr) for lr in state_lrs])
        plt.colorbar(cmap, ax=ax, label="IS likelihood ratio", orientation="vertical")

    def generate_trajectories(self, policy, num_trajectories=100, num_trials=None, pbar_kwargs=None, seed=2020, output_opt=False):
        total_num_trajectories = num_trajectories if num_trials is None else num_trajectories * num_trials

        history = BatchOfTrajectories()
        for traj_num in trange(total_num_trajectories, **pbar_kwargs):
            np.random.seed(seed + traj_num)
            trajectory = []
            s = self.initial_state
            for t in range(self.max_time):
                # Check if terminal state (discrete)
                if s in self.terminal_states:
                    break
                probs_action = policy.probs[s]
                a = np.random.choice(self.num_actions, p=probs_action)
                a_opt = policy.actions_opt[s]
                probs_transition = self.T[s, a]
                s_next = np.random.choice(self.num_states, p=probs_transition)
                r = self.R[s, a, s_next]
                trajectory.append((np.array([s]), a, r, a_opt))
                s = s_next
            history.append(*zip(*trajectory), traj_len=len(trajectory), state_term=s)

        if num_trials is not None:
            return history.reshape(num_trials, num_trajectories)
        return history