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

# Create a data function (and function space)
data_function = Function(Model(Nx=4096).function_space)

# Initialize our numerical model
model = Model(Nx=240)
obs_func = Function(model.function_space)

networks = []
for name in args.names:
    experiment = Experiment(name)
    net = experiment.NN
    if hasattr(experiment, "torch") and experiment.torch:
        # Convert to ANN
        net = ANN([1, 30, 1], bias=[True, True], sigma=ufl.tanh)
        i = 0
        j = "weight"
        for v in experiment.NN[1].state_dict().values():
            net.weights[i][j].assign(Constant(v.detach().numpy()))
            if j == "weight":
                j = "bias"
            else:
                j = "weight"
                i += 1
    elif hasattr(experiment, "fem") and experiment.fem:
        approximator = Function(model.function_space)
        approximator.vector()[:] = experiment.NN
        net_ = lambda approximator, *args: approximator
        net = partial(net_, approximator)
    networks.append(net)

t0, t1 = (0., 0.1)
dt = 0.0025
precision = 4
sampler_dt = 0.02

# Load data and subsample
dataset = Dataset("data/data.xdmf", t0, t1, 0.0001, precision, obs_func=data_function)
sampler = DeterministicSubsampler(dataset, t0, t1, sampler_dt, obs_func, keep=1.0)

x, *_ = SpatialCoordinate(model.mesh)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure()

for net in networks:
    proj_net = project(net(x), model.function_space)
    plot(proj_net)

proj_gt = project(model.ground_truth(x), model.function_space)
plot(proj_gt)

auto_adjust_limits()

ax = plt.gca()
line = ax.lines[-1]
line.set_dashes((3, 2))

legends = args.legends
legends += ["Reference"]
plt.legend(legends)
plt.xlabel("x")

plt.tight_layout()

plt.savefig("plots/plot_all.png", dpi=400, bbox_inches="tight")
