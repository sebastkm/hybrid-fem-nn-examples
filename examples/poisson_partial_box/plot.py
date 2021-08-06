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

fig = plt.figure(figsize=(7.0, 3), constrained_layout=True)

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )


# Compute limits
lower = min(proj_net.vector().min(), proj_gt.vector().min())
upper = max(proj_net.vector().max(), proj_gt.vector().max())

cmap = mpl.cm.get_cmap('viridis')
normalizer = mpl.colors.Normalize(lower, upper)
colors = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)

plt.sca(grid[0])
plot(proj_net, cmap=cmap, norm=normalizer)
# Draw subdomain rectangle.
plt.plot([0.25, 0.25, 0.75, 0.75, 0.25], [0.35, 0.65, 0.65, 0.35, 0.35], "r")

plt.sca(grid[1])
plot(proj_gt, cmap=cmap, norm=normalizer)

grid[-1].cax.colorbar(colors)
grid[-1].cax.toggle_label(True)

plt.tight_layout()
plt.savefig("plots/poisson_partial_box.png", dpi=400)
