from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed, randn
import numpy as np
from functools import partial

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.data import Dataset, DeterministicSubsampler
from shared_code.plotting import auto_adjust_limits

from torch_model import TorchModel
from model import Model
from functional import Functional

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--names", "-n", type=str, nargs="+", dest="names",
                    help="Names of experiments to plot.")
parser.add_argument("--legends", type=str, nargs="+", dest="legends",
                    help="Legends on plot.")
args = parser.parse_args()

for name in args.names:
    experiment = Experiment(name)

    if hasattr(experiment, "torch_optim"):
        pde = np.array(experiment.torch_optim[0])
        bc = np.array(experiment.torch_optim[1])
        ic = np.array(experiment.torch_optim[2])

        data = np.array(experiment.torch_optim[3])
        reg = np.array(experiment.torch_optim[4])

        L_pde = pde + bc + ic
        L_pde = L_pde / L_pde[0]

        L_data = data + reg
        L_data = L_data / L_data[0]

        x = np.linspace(1, len(L_pde), len(L_pde))

        plt.loglog(x, L_pde)
        plt.loglog(x, L_data)

        plt.scatter(99999, L_pde[99999], marker="_", color="red", zorder=100)
        plt.text(99999 + 10000, L_pde[99999] + 0.00005, s="B")

        plt.scatter(99999, L_data[99999], marker="_", color="red", zorder=100)
        plt.text(99999 + 10000, L_data[99999] + 0.000005, s="B")

    else:
        dr = np.array(experiment.optimization_iteration_loss)
        dr_norm = dr / dr[0]

        x = np.linspace(1, len(dr_norm), len(dr_norm))
        plt.loglog(x, dr_norm)

        plt.scatter(191, dr_norm[191], marker="_", color="red", zorder=100)
        plt.text(191 - 0.0001, dr_norm[191] + 0.0001, s="A")


plt.gca().set_xlim(left=1)
legends = args.legends
plt.legend(legends)
plt.xlabel("Iterations")
plt.ylabel("Loss (normalized)")

plt.tight_layout()
plt.savefig("plots/plot_loss.png", dpi=400, bbox_inches="tight")
