from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed, randn

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment

from model import Model
from functional import Functional


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str, dest="name",
                    help="Names of experiments to plot.")
args = parser.parse_args()

experiment = Experiment(args.name)
net = experiment.NN
net.output_activation = None

model = Model(Nx=100, order=2)

x, y = SpatialCoordinate(model.mesh)
proj_net = project(net(x, y), model.function_space)
proj_gt = project(model.ground_truth(x, y), model.function_space)

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

for f, title, filename in zip([proj_net, proj_gt],
                              ("Estimated $\kappa$", "True $\kappa$"),
                              ("poisson_calderon_net", "poisson_calderon_gt")):
    fig = plt.figure(figsize=(4.0, 3), constrained_layout=True)

    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 1),
                     axes_pad=0.35,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )

    lower = f.vector().min()
    upper = f.vector().max()

    plt.sca(grid[0])
    colors = plot(f, vmin=lower, vmax=upper)
    plt.title(title)

    grid[-1].cax.colorbar(colors)
    grid[-1].cax.toggle_label(True)

    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png", dpi=400)
