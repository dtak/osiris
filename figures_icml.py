import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox
from src.environments.cases import make_env
from tqdm import tqdm, trange
from src.ope.BaseIS import IS, WIS
from src.ope.PHIS import PHWIS
from src.ope.OSIRIS import OSIRIS, OSIRWIS, OSIRWIS_nn
from src.ope.INCRIS import INCRIS
from src.ope.MarginalizedIS import MarginalizedIS
from src.utils import BatchOfTrajectories, make_dir
from src.environments.gridworld.gridworld import GridWorld

num_trials = 200
vals_num_trajectories = np.array((25, 50, 100, 250, 500))
vals_alpha = np.array((0.05, 0.1, 0.25, 0.5, 1.))
cmap = plt.cm.get_cmap('viridis')
colors = np.append(
    cmap(np.linspace(0, 0.8, num=len(vals_alpha) - 1)),
    ((0., 0., 0., 1.),), axis=0)
seed = 2020
figsave_dir = "./output/figures/"
make_dir(figsave_dir)
datasave_dir = "./output/data/"
make_dir(datasave_dir)
ESTIMATORS_TO_RUN = ("IS", "WIS", "PHWIS", "INCRIS", "MarginalizedIS", "OSIRIS", "OSIRWIS", "OSIRIS-fancyA", "OSIRWIS-fancyA", "OSIRIS-smirnov", "OSIRWIS-smirnov", "OSIRIS-nn", "OSIRWIS-nn", "OSIRIS-oracle", "OSIRWIS-oracle", "MC")
ESTIMATORS_FOR_TABLE = {"IS": "IS", "WIS": "WIS", "PHWIS": "PHWIS", "INCRIS": "INCRIS", "MarginalizedIS": "MIS", "OSIRIS": "OSIRIS", "OSIRWIS": "OSIRWIS", "MC": "On-Policy"}
ESTIMATORS_FOR_SUPP_TABLE = {
    "OSIRIS": ("Algorithm 1", "osiris"),
    "OSIRWIS": ("Algorithm 1", "osirwis"),
    "OSIRIS-smirnov": ("Smirnov", "osiris"),
    "OSIRWIS-smirnov": ("Smirnov", "osirwis"),
    "OSIRIS-fancyA": ("$g(\\tau)$-Binary $\mathcal{{A}}$", "osiris"),
    "OSIRWIS-fancyA": ("$g(\\tau)$-Binary $\mathcal{{A}}$", "osirwis"),
    "OSIRIS-nn": ("NN as $\hat{{Q}}^{{\\pi_e}}$", "osiris"),
    "OSIRWIS-nn": ("NN as $\hat{{Q}}^{{\\pi_e}}$", "osirwis"),
    "MC": ("On-Policy", "")
    }
ESTIMATORS_FOR_SUPP_TABLE_ORACLE = {
    "OSIRIS-oracle": ("Oracle", "osiris"),
    "OSIRWIS-oracle": ("Oracle", "osirwis"),
    }
ORACLE_ESTIMATORS = ("OSIRIS-oracle", "OSIRWIS-oracle")
ENVIRONMENTS_FOR_TABLE = {
    "GridworldDD": "Dilly-Dallying Gridworld",
    "GridworldXP": "Express Gridworld",
    "CartPole": "Cart Pole",
    "LunarLander": "Lunar Lander",
}
ENVIRONMENTS_SUBFIGURES = {
    "GridworldDD": "a",
    "GridworldXP": "supp",
    "LunarLander": "c",
    "CartPole": "b",
}
gw_env = make_env("GridworldDD")[0]
GRIDWORLD_ORACLE = np.ones(gw_env.num_states, dtype=bool)
for rc in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1), (3, 1)]:
    GRIDWORLD_ORACLE[gw_env.maze.flatten_index(rc)] = False
ORACLE = {
    "GridworldDD": GRIDWORLD_ORACLE,
    "GridworldXP": GRIDWORLD_ORACLE,
}
environment_fnames = {
    "GridworldDD": "GridworldDD",
    "GridworldXP": "GridworldXP",
    "LunarLander": "LunarLander",
    "CartPole": "CartPole",
}
vals_state_inspect_range = {
    "GridworldDD": np.arange(98 + 1),
    "GridworldXP": np.arange(98 + 1),
    "LunarLander": np.linspace(-3, 3, num=8 + 1), # y velocity
    "CartPole": np.linspace(-1, 1, num=8 + 1), # angular velocity
}
vals_state_dim_inspect = {
    "GridworldDD": 0,
    "GridworldXP": 0,
    "LunarLander": 3, # y velocity
    "CartPole": 3, # angular velocity
}
AXHLINE_WIDTH = 0.75
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["axes.labelpad"] = 0.8
plt.rcParams["lines.linewidth"] = 0.9
plt.rcParams["lines.markersize"] = 3
plt.rcParams['boxplot.meanprops.linewidth'] = 0.9
plt.rcParams['boxplot.whiskerprops.linewidth'] = 0.75
plt.rcParams['boxplot.capprops.linewidth'] = 0.75
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['legend.markerscale'] = 0.9      # the relative size of legend markers vs. original
plt.rcParams['legend.fontsize'] = 5.5
plt.rcParams['legend.labelspacing'] = 0.2  # the vertical space between the legend entries
plt.rcParams['legend.handlelength'] = 1.5  # the length of the legend lines
plt.rcParams['legend.handletextpad'] = 0.4  # the space between the legend line and legend text
FIG1_W = 3.25 # inches
FIG1_HEIGHT = 0.802125 # inches
FIG1_PAD = 0.05 # inches
FIG1_CBAR_HEIGHT = 0.4 # inches
FIG2_W = 2.12 # inches
FIG2_H = 1.3 # inches
FIG2_PADR = 0.05 # inches
FIG2_PADT = 0.02 # inches
FIG2_PADL = 0.54 # inches
FIG2_PADB = 0.31 # inches
FIG3_W = 2.12 # inches
FIG3_H = 1.06 # inches
FIG3_BOTTOM = 0.3008 # inches
FIG3_LEFT = 0.5512 # inches
FIG3_HIST_WIDTH = 0.2756 # inches
FIG3_HIST_PAD = 0.0224 # inches
FIG4bc_W = 1.6875 # inches
FIG4bc_TOP = 0.3 # inches
FIG4a_LEFT = 0.36 # inches
FIG4bc_LEFT = 0.405 # inches
FIG4_BOTTOM = 0.05 # inches
FIG4_RIGHT = 0.05 # inches
FIG4a_CBAR_WIDTH = 0.1 # inches
FIG4a_CBAR_PAD = 0.05 # inches
FIG1_H = FIG1_HEIGHT + FIG1_PAD * 2
FIG3_SCAT_HEIGHT = FIG3_H - FIG3_BOTTOM - FIG3_HIST_WIDTH - FIG3_HIST_PAD * 2
FIG3_SCAT_WIDTH = FIG3_W - FIG3_LEFT - FIG3_HIST_WIDTH - FIG3_HIST_PAD * 2
FIG4bc_AX_WIDTH = FIG4bc_W - FIG4bc_LEFT - FIG4_RIGHT
FIG4_H = FIG4_BOTTOM + FIG1_HEIGHT + FIG4bc_TOP
FIG4a_MAP_WIDTH = FIG1_W - FIG4a_LEFT - FIG4a_CBAR_WIDTH - FIG4a_CBAR_PAD - FIG4_RIGHT

table_results = {}
table_results_supp = {}
table_results_supp_oracle = {}
for environment_id, environment_fname in tqdm(environment_fnames.items(), position=0, desc="Environment"):
    # Make environment
    env, pi_e, pi_b, num_trajectories, discount, state_discretizer = make_env(environment_id)
    if num_trajectories not in vals_num_trajectories:
        vals_num_trajectories = vals_num_trajectories.append(num_trajectories)
    max_num_trajectories = np.max(vals_num_trajectories)
    idx_alpha_OSIRIS = np.argmax(vals_alpha == 0.05)
    idx_num_trajectories = np.argmax(vals_num_trajectories == num_trajectories)

    # Initialize
    fname_estimates_baselines = datasave_dir + "estimates_baselines_{:s}.npz".format(environment_id)
    if os.path.isfile(fname_estimates_baselines):
        loaded_file = np.load(fname_estimates_baselines)
        assert len(loaded_file["estimates"].shape) == 2
        assert loaded_file["estimates"].shape[1] == num_trials
        estimators_in_file = list(loaded_file["estimators"])
        estimates_in_file = loaded_file["estimates"]
    else:
        estimators_in_file = []
        estimates_in_file = None
    calc_estimates_baselines = {estimator: estimator not in estimators_in_file for estimator in ESTIMATORS_TO_RUN}
    estimates = {estimator: estimates_in_file[estimators_in_file.index(estimator)] if not calc_estimates_baselines[estimator] else np.empty(num_trials) for estimator in ESTIMATORS_TO_RUN}
    fname_estimates_OSIRIS = datasave_dir + "estimates_OSIRIS_{:s}.npz".format(environment_id)
    shape_estimates_OSIRIS = (len(vals_alpha), len(vals_num_trajectories), num_trials)
    if os.path.isfile(fname_estimates_OSIRIS):
        loaded_file = np.load(fname_estimates_OSIRIS)
        assert np.all(vals_alpha == loaded_file["vals_alpha"])
        assert np.all(vals_num_trajectories == loaded_file["vals_num_trajectories"])
        assert loaded_file["estimates_OSIRIS"].shape == shape_estimates_OSIRIS
        assert loaded_file["estimates_OSIRWIS"].shape == shape_estimates_OSIRIS
        estimates_OSIRIS = loaded_file["estimates_OSIRIS"]
        estimates_OSIRWIS = loaded_file["estimates_OSIRWIS"]
        calc_estimates_OSIRIS = False
    else:
        estimates_OSIRIS = np.empty(shape_estimates_OSIRIS)
        estimates_OSIRWIS = np.empty(shape_estimates_OSIRIS)
        calc_estimates_OSIRIS = True
    fname_scatter_OSIRIS = datasave_dir + "scatter_OSIRIS_{:s}.npz".format(environment_id)
    shape_scatter_OSIRIS = (len(vals_alpha), num_trials, num_trajectories)
    if os.path.isfile(fname_scatter_OSIRIS):
        loaded_file = np.load(fname_scatter_OSIRIS)
        assert np.all(vals_alpha == loaded_file["vals_alpha"])
        assert loaded_file["scatter_wts"].shape == shape_scatter_OSIRIS
        assert loaded_file["scatter_len"].shape == shape_scatter_OSIRIS
        scatter_wts = loaded_file["scatter_wts"]
        scatter_len = loaded_file["scatter_len"]
        calc_scatter_OSIRIS = False
    else:
        scatter_wts = np.empty(shape_scatter_OSIRIS)
        scatter_len = np.empty(shape_scatter_OSIRIS)
        calc_scatter_OSIRIS = True

    estimates_OSIRIS_oracle = np.empty(num_trials)
    estimates_OSIRWIS_oracle = np.empty(num_trials)

    states_inspect = []
    correls_inspect = []

    # Do calculations on behavior trajectories
    trajsave_dir = datasave_dir + "history_b_{:s}/".format(environment_id)
    make_dir(trajsave_dir)
    precalcsave_dir = datasave_dir + "precalc_OSIRIS_{:s}/".format(environment_id)
    make_dir(precalcsave_dir)
    for trial_num in trange(num_trials, desc="Trials (calculate beh)", position=1, leave=False):
        # Pre-generate/load trajectories
        fname = trajsave_dir + "seed_{:d}.npz".format(seed + trial_num)
        if os.path.isfile(fname):
            history_b = BatchOfTrajectories().load_npz(fname)
        else:
            history_b = env.generate_trajectories(pi_b, num_trajectories=max_num_trajectories, seed=seed+trial_num, pbar_kwargs={"desc": "Trajectories", "position": 2, "leave": False})
            history_b.save_npz(fname)
        history_b.precalculate(pi_e, pi_b)
        # Do calculations
        fname = precalcsave_dir + "seed_{:d}.npz".format(seed + trial_num)
        if os.path.isfile(fname):
            precalc_OSIRIS = dict(np.load(fname))
            need_to_save = False
        else:
            precalc_OSIRIS = {}
            need_to_save = True
        for idx_num_traj, num_traj in enumerate(vals_num_trajectories):
            _history_b = history_b.reshape(1, num_traj)[0]
            # Pre-calculate/load states_discrete and correls_map
            states_discrete = precalc_OSIRIS.get("states_discrete_{:d}".format(num_traj))
            correls_map = precalc_OSIRIS.get("correls_map_{:d}".format(num_traj))
            correls_map_fancy = precalc_OSIRIS.get("correls_map_fancy_{:d}".format(num_traj))
            correls_map_smirnov = precalc_OSIRIS.get("correls_map_smirnov_{:d}".format(num_traj))
            discretized = _history_b.precalculate_OSIRIS(states_discrete, correls_map, correls_map_fancy, correls_map_smirnov, state_discretizer)
            if discretized and states_discrete is None:
                precalc_OSIRIS["states_discrete_{:d}".format(num_traj)] = np.concatenate(_history_b.states_discrete, axis=0)
                need_to_save = True
            if correls_map is None:
                precalc_OSIRIS["correls_map_{:d}".format(num_traj)] = _history_b.correls_map
                need_to_save = True
            if correls_map_fancy is None:
                precalc_OSIRIS["correls_map_fancy_{:d}".format(num_traj)] = _history_b.correls_map_fancy
                need_to_save = True
            if correls_map_smirnov is None:
                precalc_OSIRIS["correls_map_smirnov_{:d}".format(num_traj)] = _history_b.correls_map_smirnov
                need_to_save = True
            # Run OSIRIS for different alphas
            if calc_estimates_OSIRIS:
                for idx_alph, alph in enumerate(vals_alpha):
                    estimates_OSIRIS[idx_alph][idx_num_traj][trial_num] = OSIRIS(_history_b, alpha=alph)
                    estimates_OSIRWIS[idx_alph][idx_num_traj][trial_num] = OSIRWIS(_history_b, alpha=alph)
            if num_traj == num_trajectories:
                # Track inspect states
                states_inspect.append(np.concatenate([s for s in _history_b.states]))
                correls_inspect.append(np.concatenate([_history_b.correls_map[s] for s in _history_b.states_discrete]))
                # Run estimators
                if calc_estimates_baselines["IS"]:
                    estimates["IS"][trial_num] = IS(_history_b)
                if calc_estimates_baselines["WIS"]:
                    estimates["WIS"][trial_num] = WIS(_history_b)
                if calc_estimates_baselines["PHWIS"]:
                    estimates["PHWIS"][trial_num] = PHWIS(_history_b)
                if calc_estimates_baselines["INCRIS"]:
                    estimates["INCRIS"][trial_num] = INCRIS(_history_b)
                if calc_estimates_baselines["MarginalizedIS"]:
                    estimates["MarginalizedIS"][trial_num] = MarginalizedIS(_history_b)
                if calc_estimates_baselines["OSIRIS"]:
                    estimates["OSIRIS"][trial_num] = estimates_OSIRIS[idx_alpha_OSIRIS][idx_num_trajectories][trial_num]
                if calc_estimates_baselines["OSIRWIS"]:
                    estimates["OSIRWIS"][trial_num] = estimates_OSIRWIS[idx_alpha_OSIRIS][idx_num_trajectories][trial_num]
                if calc_estimates_baselines["OSIRIS-fancyA"]:
                    estimates["OSIRIS-fancyA"][trial_num] = OSIRIS(_history_b, alpha=vals_alpha[idx_alpha_OSIRIS], mod="fancyA")
                if calc_estimates_baselines["OSIRWIS-fancyA"]:
                    estimates["OSIRWIS-fancyA"][trial_num] = OSIRWIS(_history_b, alpha=vals_alpha[idx_alpha_OSIRIS], mod="fancyA")
                if calc_estimates_baselines["OSIRIS-smirnov"]:
                    estimates["OSIRIS-smirnov"][trial_num] = OSIRIS(_history_b, alpha=0.2, mod="smirnov")
                if calc_estimates_baselines["OSIRWIS-smirnov"]:
                    estimates["OSIRWIS-smirnov"][trial_num] = OSIRWIS(_history_b, alpha=0.2, mod="smirnov")
                if calc_estimates_baselines["OSIRIS-nn"]:
                    estimates["OSIRIS-nn"][trial_num], estimates["OSIRWIS-nn"][trial_num] = OSIRWIS_nn(_history_b)
                if calc_estimates_baselines["OSIRIS-oracle"] and environment_id in ORACLE:
                    estimates_OSIRIS_oracle[trial_num] = OSIRIS(_history_b, alpha=vals_alpha[idx_alpha_OSIRIS], keep_map=ORACLE[environment_id])
                if calc_estimates_baselines["OSIRWIS-oracle"] and environment_id in ORACLE:
                    estimates_OSIRWIS_oracle[trial_num] = OSIRWIS(_history_b, alpha=vals_alpha[idx_alpha_OSIRIS], keep_map=ORACLE[environment_id])
                if calc_scatter_OSIRIS:
                    for idx_alph, alph in enumerate(vals_alpha):
                        keep_map = _history_b.correls_map <= alph
                        theta_t = [keep_map[s_disc] for s_disc in _history_b.states_discrete]
                        scatter_len[idx_alph, trial_num] = [np.sum(x) for x in theta_t]
                        scatter_wts[idx_alph, trial_num] = [np.prod(p[x]) for p, x in zip(_history_b.rho_ts, theta_t)]
        if need_to_save:
            np.savez_compressed(fname, **precalc_OSIRIS)
    if calc_estimates_baselines["OSIRIS-oracle"] and environment_id in ORACLE: # else it'll be empty from initialization above
        estimates["OSIRIS-oracle"] = estimates_OSIRIS_oracle
    if calc_estimates_baselines["OSIRWIS-oracle"] and environment_id in ORACLE:
        estimates["OSIRWIS-oracle"] = estimates_OSIRWIS_oracle
    if calc_estimates_OSIRIS:
        np.savez_compressed(fname_estimates_OSIRIS, estimates_OSIRIS=estimates_OSIRIS, estimates_OSIRWIS=estimates_OSIRWIS, vals_alpha=vals_alpha, vals_num_trajectories=vals_num_trajectories)
    if calc_scatter_OSIRIS:
        np.savez_compressed(fname_scatter_OSIRIS, scatter_wts=scatter_wts, scatter_len=scatter_len, vals_alpha=vals_alpha)
    states_inspect = np.concatenate(states_inspect)
    correls_inspect = np.concatenate(correls_inspect)

    # Pre-generate/load evaluation trajectory returns
    fname = datasave_dir + "history_e_{:s}_seed_{:d}.npz".format(environment_id, seed)
    if os.path.isfile(fname):
        returns_e = np.load(fname)["arr_0"]
    else:
        returns_e = np.empty((num_trials, num_trajectories))
        pbar_kwargs = {"desc": "Trajectories", "position": 2, "leave": False}
        for trial_num in trange(num_trials, desc="Trials (load eval traj)", position=1, leave=False):
            history_e = env.generate_trajectories(pi_e, num_trajectories=num_trajectories, seed=seed+trial_num, pbar_kwargs=pbar_kwargs)
            returns_e[trial_num] = np.array([r.sum() for r in history_e.rewards])
        np.savez(fname, returns_e) # num_trials x num_trajectories
    true_val = np.mean(returns_e)
    if calc_estimates_baselines["MC"]:
        estimates["MC"] = np.mean(returns_e, axis=1) # mean over trajectories

    if any(calc_estimates_baselines.values()):
        np.savez_compressed(fname_estimates_baselines, estimates=np.array([estimates[estimator] for estimator in ESTIMATORS_TO_RUN]), estimators=ESTIMATORS_TO_RUN)

    # Make Tab1
    performance_means = {}
    performance_stds = {}
    performance_rmses = {}
    for estimator, est in estimates.items():
        all_estimates = np.array(est)
        performance_means[estimator] = all_estimates.mean()
        performance_stds[estimator] = all_estimates.std()
        performance_rmses[estimator] = np.sqrt(np.mean((all_estimates - true_val) ** 2))
        print(estimator, performance_means[estimator], performance_stds[estimator], performance_rmses[estimator])
    def make_str(data, data_sort, est):
        if data_sort:
            min_val = min(data_sort[estimator] for estimator in est.keys())
        else:
            min_val = None
        def _result_fmt(estimator):
            if estimator in ORACLE_ESTIMATORS and environment_id not in ORACLE:
                return ""
            elif min_val and data_sort[estimator] == min_val:
                return "$\mathbf{{{:.1f}}}$"
            else:
                return "${:.1f}$"
        return " & ".join(_result_fmt(estimator).format(data[estimator]) for estimator in est.keys())
    TABLE_RESULTS_FORMAT = """\\multirow{{3}}{{{width:s}}}{{\\textbf{{{env_name:s}}}}}
    & Mean & {means:s} \\\\
    & Std  & {stds:s} \\\\
    & RMSE & {rmses:s} \\\\"""
    table_results[environment_id] = TABLE_RESULTS_FORMAT.format(
        width="1in",
        env_name=ENVIRONMENTS_FOR_TABLE[environment_id],
        means=make_str(performance_means, {estimator: np.abs(est - true_val) if not estimator == "MC" and not estimator in ORACLE_ESTIMATORS else float("inf") for estimator, est in performance_means.items()}, ESTIMATORS_FOR_TABLE),
        stds=make_str(performance_stds, {estimator: est if not estimator == "MC" and not estimator in ORACLE_ESTIMATORS else float("inf") for estimator, est in performance_stds.items()}, ESTIMATORS_FOR_TABLE),
        rmses=make_str(performance_rmses, {estimator: est if not estimator == "MC" and not estimator in ORACLE_ESTIMATORS else float("inf") for estimator, est in performance_rmses.items()}, ESTIMATORS_FOR_TABLE))
    table_results_supp[environment_id] = TABLE_RESULTS_FORMAT.format(
        width="0.75in",
        env_name=ENVIRONMENTS_FOR_TABLE[environment_id],
        means=make_str(performance_means, {estimator: np.abs(est - true_val) if not estimator == "MC" and not estimator in ORACLE_ESTIMATORS else float("inf") for estimator, est in performance_means.items()}, ESTIMATORS_FOR_SUPP_TABLE),
        stds=make_str(performance_stds, {estimator: est if not estimator == "MC" and not estimator in ORACLE_ESTIMATORS else float("inf") for estimator, est in performance_stds.items()}, ESTIMATORS_FOR_SUPP_TABLE),
        rmses=make_str(performance_rmses, {estimator: est if not estimator == "MC" and not estimator in ORACLE_ESTIMATORS else float("inf") for estimator, est in performance_rmses.items()}, ESTIMATORS_FOR_SUPP_TABLE))
    if environment_id in ORACLE:
        table_results_supp_oracle[environment_id] = TABLE_RESULTS_FORMAT.format(
            width="0.75in",
            env_name=ENVIRONMENTS_FOR_TABLE[environment_id],
            means=make_str(performance_means, None, ESTIMATORS_FOR_SUPP_TABLE_ORACLE),
            stds=make_str(performance_stds, None, ESTIMATORS_FOR_SUPP_TABLE_ORACLE),
            rmses=make_str(performance_rmses, None, ESTIMATORS_FOR_SUPP_TABLE_ORACLE))

    # Make Fig1 - Plot gridworld environment
    if environment_id == "GridworldXP":
        fig = plt.figure(figsize=(FIG1_W, FIG4_H))
        ax_cbar = fig.add_axes([(FIG1_PAD + FIG4a_MAP_WIDTH + FIG4a_CBAR_PAD) / FIG1_W, FIG1_PAD / FIG1_H, FIG4a_CBAR_WIDTH / FIG1_W, FIG1_CBAR_HEIGHT / FIG1_H])
        ax_map = fig.add_axes([FIG1_PAD / FIG1_W, FIG1_PAD / FIG1_H, FIG4a_MAP_WIDTH / FIG1_W, FIG1_HEIGHT / FIG1_H])
        row_count, col_count = env.maze_dimensions
        maze_dims = (row_count, col_count)
        rewards = np.zeros(maze_dims)
        wall_info = .5 + np.zeros(maze_dims)
        wall_mask = np.zeros(maze_dims)
        for row in range(row_count):
            for col in range(col_count):
                if env.maze.topology[row][col] == '#':
                    wall_mask[row,col] = 1
                rewards[row,col] = env.rewards.get(env.maze.topology[row][col], 0) + env.rewards.get("moved", 0) # assume successfully moved
        wall_info = np.ma.masked_where(wall_mask==0, wall_info)
        rewards = np.ma.masked_array(rewards, mask=np.logical_or(wall_mask, rewards == 0))
        cmap = plt.cm.get_cmap("winter", 2)(np.arange(2))
        cmap = np.insert(cmap, 1, [[1, 1, 1, 1]], axis=0)
        cmap = ListedColormap(cmap)
        rewards_plot = ax_map.pcolormesh(np.arange(-0.5, col_count), np.arange(-0.5, row_count), rewards, cmap=cmap)
        ax_map.pcolormesh(np.arange(-0.5, col_count), np.arange(-0.5, row_count), wall_info, cmap='gray')
        y,x = env.maze.start_coords
        ax_map.text(x,y,'start', color='gray', fontsize=5, va='center', ha='center', fontweight='bold')
        for row in range(row_count):
            for col in range(col_count):
                if env.maze.topology[row][col] in env.terminal_markers:
                    y,x = row,col
                    ax_map.text(x,y,'end', color='gray', fontsize=5, va='center', ha='center', fontweight='bold')
        # Show only half of the border wall states
        nonwall_x = np.nonzero(np.any(wall_mask == 0, axis=0))[0]
        nonwall_y = np.nonzero(np.any(wall_mask == 0, axis=1))[0]
        ax_map.set_xlim(left=nonwall_x[0] - 1, right=nonwall_x[-1] + 1)
        ax_map.set_ylim(top=nonwall_y[0] - 1, bottom=nonwall_y[-1] + 1)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        plt.colorbar(rewards_plot, cax=ax_cbar, label='Reward', orientation='vertical', ticks=[-10/3, 0, 10/3])
        ax_cbar.set_yticklabels(["$-5$", "$0$", "$+5$"])

        for row in range( row_count ):
            for col in range( col_count ):
                if wall_mask[row][col] == 1 or env.maze.get_unflat((row, col)) in env.terminal_markers:
                    continue
                probs_from_state = pi_b.get_probs(env.maze.flatten_index((row, col)))
                for a, prob in enumerate(probs_from_state):
                    if prob > 0:
                        dy, dx = 0.5 * env.actions[a] * prob
                        alpha = 0.2 + 0.6 * prob / probs_from_state.max() # normalize to [0.2, 0.8] so that everything is still visible
                        c = 'r' if prob in (0.125, 0.625) else 'b'
                        ax_map.arrow(col, row, dx, dy,
                            shape='full', facecolor=c, edgecolor=c, linewidth=0.5, length_includes_head=False, head_width=.1, alpha=alpha)

        plt.savefig(figsave_dir + "fig1_gridworld.pdf")
        plt.close()

    # Make Fig2 - Change parameter experiments
    def make_experiment_plot(fname, fname_legend, xlabel, true_val, data_x, data_ests):

        xcenters = np.arange(len(data_x)) * 0.3 * len(data_ests)
        xoffsets = np.arange(len(data_ests)) * 0.3
        xoffsets -= np.mean(xoffsets)

        fig = plt.figure(figsize=(FIG2_W, FIG2_H))
        ax2 = fig.add_axes([FIG2_PADL / FIG2_W, FIG2_PADB / FIG2_H, (FIG2_W - FIG2_PADR - FIG2_PADL) / FIG2_W, (FIG2_H - FIG2_PADT - FIG2_PADB) / FIG2_H])
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("OPE value estimate")
        lines = []
        for i, (data, xoffset, color) in enumerate(zip(data_ests, xoffsets, colors)):
            ln = ax2.errorbar(xcenters + xoffset, [np.mean(x) for x in data], yerr=[np.std(x) for x in data], c=color, marker="o")
            lines.append(ln[0])
        ax2.axhline(true_val, color="k", linewidth=AXHLINE_WIDTH)
        ax2.set_xticks(xcenters)
        ax2.set_xticklabels(data_x)
        ax2.secondary_yaxis("right").tick_params(axis="y", direction="in", labelright=False)
        plt.savefig(fname)
        plt.close()

        plt.figure(figsize=(0.39, 0.7))
        ax3 = plt.gca()
        ax3.axis("off")
        ax3.legend(lines, vals_alpha, title=r"$\alpha$", loc="center")
        plt.savefig(fname_legend)
        plt.close()
    if environment_id in ENVIRONMENTS_SUBFIGURES:
        make_experiment_plot(
            fname=(figsave_dir + "fig2{:s}_consistency_WIS_{:s}.pdf".format(ENVIRONMENTS_SUBFIGURES[environment_id], environment_fname)),
            fname_legend=(figsave_dir + "fig2_legend.pdf"),
            xlabel=r'Number of trajectories $|\mathcal{D}|$',
            true_val=true_val,
            data_x=vals_num_trajectories,
            data_ests=estimates_OSIRWIS
            )

    # Make Fig3 - scatter plot traj len x traj weight
    if environment_id in ENVIRONMENTS_SUBFIGURES:
        NUM_SCATTER_PTS = 150
        fig = plt.figure(figsize=(FIG3_W, FIG3_H))
        plt_scat = fig.add_subplot(2, 2, 3)
        plt_lenhist = fig.add_subplot(2, 2, 1, sharex=plt_scat)
        plt_wthist = fig.add_subplot(2, 2, 4, sharey=plt_scat)
        plt_scat.set_position([FIG3_LEFT / FIG3_W, FIG3_BOTTOM / FIG3_H, FIG3_SCAT_WIDTH / FIG3_W, FIG3_SCAT_HEIGHT / FIG3_H])
        plt_lenhist.set_position([FIG3_LEFT / FIG3_W, (FIG3_BOTTOM + FIG3_SCAT_HEIGHT + FIG3_HIST_PAD) / FIG3_H, FIG3_SCAT_WIDTH / FIG3_W, FIG3_HIST_WIDTH / FIG3_H])
        plt_wthist.set_position([(FIG3_LEFT + FIG3_SCAT_WIDTH + FIG3_HIST_PAD) / FIG3_W, FIG3_BOTTOM / FIG3_H, FIG3_HIST_WIDTH / FIG3_W, FIG3_SCAT_HEIGHT / FIG3_H])
        scat_hdls = []
        log_scatter_wts = np.log10(scatter_wts)
        for _data_lens, _data_wts, _alpha, _color in zip(reversed(scatter_len), reversed(scatter_wts), reversed(vals_alpha), reversed(colors)):
            h = plt_scat.scatter(_data_lens.flatten()[:NUM_SCATTER_PTS], _data_wts.flatten()[:NUM_SCATTER_PTS],
                                    marker="x", alpha=0.25, color=_color)
            scat_hdls.append(h)
        bp_len = plt_lenhist.boxplot(scatter_len.reshape(len(vals_alpha), -1).T, vert=False, showfliers=False, widths=0.7)
        bp_wts = plt_wthist.boxplot(scatter_wts.reshape(len(vals_alpha), -1).T, vert=True, showfliers=False, widths=0.7)
        for idx, _color in enumerate(colors):
            bp_len["boxes"][idx].set_color(_color)
            bp_wts["boxes"][idx].set_color(_color)
            bp_len["medians"][idx].set_color(_color)
            bp_wts["medians"][idx].set_color(_color)
        plt_scat.set_yscale("log")
        plt_wthist.set_yscale("log")
        plt_scat.yaxis.set_major_locator(LogLocator(numticks=4))
        plt_scat.tick_params(axis="y", which="minor", left=True)
        plt_scat.set_yticklabels([], minor=True)
        plt_scat.set_ylabel("OSIRIS weight")
        plt_scat.set_xlabel("Effective trajectory length")
        plt_lenhist.axis("off")
        plt_wthist.axis("off")
        plt.savefig(figsave_dir + "fig3{:s}_scatter_{:s}.pdf".format(ENVIRONMENTS_SUBFIGURES[environment_id], environment_fname))
        plt.close()

        plt.figure(figsize=(0.39, 0.7))
        plt.axis("off")
        leg = plt.legend(handles=reversed(scat_hdls), labels=list(vals_alpha), title=r"$\alpha$", loc="center")
        for leg_hdl in leg.legendHandles:
            leg_hdl.set_alpha(1.)
        plt.savefig(figsave_dir + "fig3_legend.pdf")
        plt.close()

    # # Make Fig4 - state relevance interpretation
    if environment_id in ENVIRONMENTS_SUBFIGURES:
        def plot_for_dim(dim):
            states_slice_inspect = states_inspect[:, dim]
            relevant_inspect = correls_inspect <= vals_alpha[idx_alpha_OSIRIS]
            bin_edges = vals_state_inspect_range[environment_id]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts_all, _ = np.histogram(states_slice_inspect, bins=bin_edges)
            counts_kept, _ = np.histogram(states_slice_inspect[relevant_inspect], bins=bin_edges)
            est_relevance = np.ma.masked_array(counts_kept / counts_all, mask=(counts_all == 0))
            if type(env) is GridWorld:
                fig = plt.figure(figsize=(FIG1_W, FIG4_H))
                ax_cbar = fig.add_axes([FIG4a_LEFT / FIG1_W, FIG4_BOTTOM / FIG4_H, FIG4a_CBAR_WIDTH / FIG1_W, FIG1_HEIGHT / FIG4_H])
                ax_map = fig.add_axes([(FIG4a_LEFT + FIG4a_CBAR_WIDTH + FIG4a_CBAR_PAD) / FIG1_W, FIG4_BOTTOM / FIG4_H, FIG4a_MAP_WIDTH / FIG1_W, FIG1_HEIGHT / FIG4_H])
                est_relevance[env.terminal_states] = np.ma.masked
                plt.sca(ax_map)
                correls_plot = env.plot(custom_data=est_relevance, custom_colorbar=True, data_norm=TwoSlopeNorm(vcenter=0.2, vmin=0., vmax=np.ma.max(est_relevance)), data_colormap="RdGy_r", fontsize_startend=6)
                plt.colorbar(correls_plot, cax=ax_cbar, orientation='vertical')
                ax_cbar.yaxis.set_ticks_position("left")
                if environment_id == "20200722_01":
                    plt.plot([10], [1], marker="*", color="g")
            else:
                fig = plt.figure(figsize=(FIG4bc_W, FIG4_H))
                ax = fig.add_axes([FIG4bc_LEFT / FIG4bc_W, FIG4_BOTTOM / FIG4_H, FIG4bc_AX_WIDTH / FIG4bc_W, FIG1_HEIGHT / FIG4_H])
                ax.plot(bin_centers, est_relevance, color="k")
                ax.set_xlabel(env.state_dim_names[dim])
                ax.xaxis.set_ticks_position("top")
                ax.xaxis.set_label_position("top")
                ax.set_ylim(bottom=0, top=est_relevance.max() * 1.1)
        for dim in range(states_inspect.shape[1]):
            plot_for_dim(dim)
            plt.savefig(figsave_dir + "fig4{:s}_relevance_{:s}_dim{:d}.pdf".format(ENVIRONMENTS_SUBFIGURES[environment_id], environment_fname, dim))
            plt.close()

table = """\\begin{{tabular}}{{ll|{cols:s}|rr|r}}
\\toprule
&& {header:s} \\\\
\\midrule
{results:s}
\\bottomrule
\\end{{tabular}}""".format(cols=("r"*(len(ESTIMATORS_FOR_TABLE) - 3)), header=" & ".join(ESTIMATORS_FOR_TABLE.values()), results="\n\\hline\n".join(table_results[env] for env in ENVIRONMENTS_FOR_TABLE.keys()))
print(table)
with open(figsave_dir + "tab1_accuracies.tex", "w") as table_file:
    table_file.write(table)
def _header_fmt(key, estimator):
    if estimator[0] == "On-Policy":
        return "\multirow{{2}}{{0.4in}}{{{:s}}}"
    return "\multicolumn{{2}}{{b{{0.8in}}}}{{{:s}}}"
table_supp = """\\begin{{tabular}}{{ll|{cols:s}|r}}
\\toprule
&& {header1:s} \\\\
&& {header2:s} \\\\
\\midrule
{results:s}
\\bottomrule
\\end{{tabular}}""".format(
    cols=("r"*(len(ESTIMATORS_FOR_SUPP_TABLE) - 1)),
    header1=" & ".join(_header_fmt(key, x).format(x[0]) for i, (key, x) in enumerate(ESTIMATORS_FOR_SUPP_TABLE.items()) if i % 2 == 0),
    header2=" & ".join(x[1] for x in ESTIMATORS_FOR_SUPP_TABLE.values()),
    results="\n\\hline\n".join(table_results_supp[env] for env in ENVIRONMENTS_FOR_TABLE.keys())
    )
print(table_supp)
with open(figsave_dir + "supp_tab1_accuracies.tex", "w") as table_file:
    table_file.write(table_supp)
table_supp_oracle = """\\begin{{tabular}}{{ll|{cols:s}}}
\\toprule
&& {header1:s} \\\\
&& {header2:s} \\\\
\\midrule
{results:s}
\\bottomrule
\\end{{tabular}}""".format(
    cols=("r"*(len(ESTIMATORS_FOR_SUPP_TABLE_ORACLE))),
    header1=" & ".join(_header_fmt(key, x).format(x[0]) for i, (key, x) in enumerate(ESTIMATORS_FOR_SUPP_TABLE_ORACLE.items()) if i % 2 == 0),
    header2=" & ".join(x[1] for x in ESTIMATORS_FOR_SUPP_TABLE_ORACLE.values()),
    results="\n\\hline\n".join(table_results_supp_oracle[env] for env in ENVIRONMENTS_FOR_TABLE.keys() if env in ORACLE)
    )
print(table_supp_oracle)
with open(figsave_dir + "supp_tab2_accuracies.tex", "w") as table_file:
    table_file.write(table_supp_oracle)
