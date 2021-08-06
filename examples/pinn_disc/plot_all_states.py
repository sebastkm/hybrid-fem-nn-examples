from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed, randn
from functools import partial

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.data import Dataset, DeterministicSubsampler
from shared_code.plotting import auto_adjust_limits

from torch_model import TorchModel
from model import Model
from functional import Functional


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--names", "-n", type=str, nargs="+", dest="names",
                    help="Names of experiments to plot.")
parser.add_argument("--legends", type=str, nargs="+", dest="legends",
                    help="Legends on plot.")
args = parser.parse_args()

data_model = Model(Nx=4096)
data_function = Function(data_model.function_space)
obs_func = Function(data_model.function_space)

model = Model(Nx=240)

networks = []
for name in args.names:
    experiment = Experiment(name)
    net = experiment.NN
    if hasattr(experiment, "torch") and experiment.torch:
        # Convert to ANN
        net = ANN([1, 30, 1], bias=[True, True], sigma=ufl.tanh)
        i = 0
        j = "weight"
        for v in experiment.NN[-1].state_dict().values():
            net.weights[i][j].assign(Constant(v.detach().numpy()))
            if j == "weight":
                j = "bias"
            else:
                j = "weight"
                i += 1
    elif hasattr(experiment, "fem") and experiment.fem:
        approximator = Function(model.function_space)
        approximator.vector()[:] = experiment.NN
        approximator = interpolate(approximator, data_model.function_space)
        net_ = lambda approximator, *args: approximator
        net = partial(net_, approximator)
    net.output_activation = None
    networks.append(net)

t0, t1 = (0., 0.1)
dt = 0.0025
precision = 4
sampler_dt = 0.02

# Load data and subsample
dataset = Dataset("data/data.xdmf", t0, t1, 0.0001, precision, obs_func=data_function)
sampler = DeterministicSubsampler(dataset, t0, t1, sampler_dt, obs_func, keep=1.0)

x, *_ = SpatialCoordinate(data_model.mesh)


class Context:
    def __init__(self, t1):
        self.u = None
        self.t1 = t1

    def solver_step(self, numerical_solution, t):
        if t >= self.t1:
            self.u = numerical_solution.copy(deepcopy=True)


errs = []
for net in networks:
    context = Context(t1)
    ic = project(data_model.initial_condition(x), data_model.function_space)
    data_model.forward(ic, t0, t1, dt, net, context)
    errs.append(assemble((context.u - sampler.get_observation(t1))**2*dx))
    plot(context.u)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import copy

obs = sampler.get_observation(t1)
plot(obs)


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""
    # Taken from https://stackoverflow.com/a/35094823

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot, top)


ax = plt.gca()
ax.set_xlim([0.2, 0.8])
autoscale_y(ax)
line = ax.lines[-1]
line.set_dashes((3, 2))

legends = args.legends
legends += ["Reference"]
plt.legend(legends)
plt.xlabel("x")

axins = inset_axes(ax, "20%", "20%", loc=2, bbox_to_anchor=(0., 0., 1.0, 1.0),
                   bbox_transform=ax.transAxes)

for line in ax.lines:
    l = axins.plot(*line.get_data())
    l[0].set_linestyle(line.get_linestyle())
    l[0].zorder = line.zorder
axins.set_xlim(0.245, 0.255)
autoscale_y(axins)
axins.set_yticks([])
axins.set_xticks([])
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.savefig("plots/plot_all_states.png", dpi=400, bbox_inches="tight")
