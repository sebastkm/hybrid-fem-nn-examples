from fenics import *
import torch
import torch.optim as optim

import argparse
import ufl
import time
from numpy.random import seed, randn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from functools import partial

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.utils import get_precision

from model import Model
from network import GTIonTerm
from functional import Functional
from data_model import DataModel
from data import Dataset, Sampler
import constants
from stimulus import get_torch_stimulus


torch.set_default_dtype(torch.float64)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--timestep", type=float, dest="dt", default=0.01,
                        help="The time step (dt) to use in the numerical model.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment.")
parser.add_argument("--time", type=float, dest="t", default=5.0,
                        help="The time to plot the state at.")
parser.add_argument("--save", type=str, default="all", dest="save",
                    help="Save all or only some snapshot. Options are: all, last or an integer."
                         "If given an integer n, only every n-th snapshot will be saved.")
parser.add_argument("--dataset", type=str, dest="dataset", default="data",
                    help="Name of dataset file excluding extension. Default: data")
parser.add_argument("--stimulus", type=int, dest="stimulus", default=0,
                    help="Stimulus ID to use. Default 0.")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="End time (T). Default is set in constants.py. Must be same as used by dataset.")
parser.add_argument("--crossed", dest="crossed", action="store_true",
                    help="Use mesh with crossed diagonals. Default is False (right diagonals).")
args = parser.parse_args()

# Create a data function (and function space)
data_model = DataModel(args.crossed)
v_func = Function(data_model.V)
s_func = Function(data_model.S)

t0, t1 = (constants.START_TIME, args.t1)
data_dt = constants.DATASET_DT
precision = constants.T_PRECISION

# Initialize our numerical model
model = Model(args.crossed)
obs_func = Function(model.V)
dt = args.dt

dataset = Dataset(f"data/{args.dataset}.xdmf", t0, t1, data_dt, precision, obs_func=(v_func, s_func))
sampler = Sampler(dataset, t0, t1, data_dt, obs_func=obs_func, s_func=Function(model.S))

stimulus = get_torch_stimulus(args.stimulus, model.V)

experiment = Experiment(args.name)
forcing, ionic_current, unet = experiment.NN
model.forcing = forcing
model.ionic_current = ionic_current
model.unet = unet


class Context:
    def __init__(self, sampler):
        self.sampler = sampler

        self.net_vs = []
        self.gt_vs = []
        self.ts = []

    def get_obs(self, t):
        return None

    def solver_step(self, v, s, t):
        obs = self.sampler.get_observation(t)
        if obs is not None:
            self.ts.append(round(float(t), constants.T_PRECISION))
            self.net_vs.append(v.detach().numpy())
            self.gt_vs.append(obs.detach().numpy())


net_v = Function(model.V)
gt_v = Function(model.V)

start_t = t0
v0 = sampler.get_observation(start_t)
ic_obs = sampler.get_ic_obs(start_t)
context = Context(sampler)

with torch.no_grad():
    model.forward(v0, start_t, args.t, dt, stimulus, ic_obs, context)

for i, t in enumerate(context.ts):
    if args.save.lower() != "all":
        if args.save.lower() == "last" and t != args.t:
            continue
        elif len(args.save.lower()) > 0 and args.save.lower()[0].isdigit():
            skips = int(args.save.lower())
            if i % skips != 0:
                continue

    net_v.vector()[:] = context.net_vs[i]
    gt_v.vector()[:] = context.gt_vs[i]

    fig = plt.figure(figsize=(7.0, 3), constrained_layout=True)

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 2),
                     axes_pad=0.35,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )

    # Compute limits
    lower = min(net_v.vector().min(), gt_v.vector().min())
    upper = max(net_v.vector().max(), gt_v.vector().max())

    cmap = mpl.cm.get_cmap('viridis')
    normalizer = mpl.colors.Normalize(lower, upper)
    colors = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)

    plt.sca(grid[0])
    plot(net_v, cmap=cmap, norm=normalizer, vmin=lower, vmax=upper)

    plt.sca(grid[1])
    plot(gt_v, cmap=cmap, norm=normalizer, vmin=lower, vmax=upper)

    grid[-1].cax.colorbar(colors)
    grid[-1].cax.toggle_label(True)

    # plt.suptitle(f"t = {t}")

    plt.tight_layout()
    plt.savefig(f"plots/paper/v/cardiac_v_stim{args.stimulus}_t{str(t).replace('.', '_')}.png", dpi=200)
    plt.close()
    print(f"Saved at {t}.")
