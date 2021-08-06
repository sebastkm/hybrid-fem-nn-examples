from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed, randn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.data import Dataset, DeterministicSubsampler
from shared_code.utils import get_precision

from model import Model, DataModel
from functional import Functional
import constants


parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", type=str, dest="name",
                    help="Name of experiment.")
args = parser.parse_args()


model = Model(Nx=constants.GROUND_TRUTH_NX)

# Construct neural network
experiment = Experiment(args.name)
net = experiment.NN
net.output_activation = None

x, y = SpatialCoordinate(model.mesh)
W = VectorFunctionSpace(model.mesh, "CG", 1)
gt_velocity = project(as_vector([model.ground_truth(Constant(0.), Constant(1.0), Constant(0.), x, y),
                                 model.ground_truth(Constant(0.), Constant(0.), Constant(1.0), x, y)]), W)
net_velocity = project(as_vector([net(Constant(0.), Constant(1.0), Constant(0.), x, y),
                                  net(Constant(0.), Constant(0.), Constant(1.0), x, y)]), W)

# A bit of a cheat in order to extract the quiver-formatted data effortlessly.
q_net = plot(net_velocity)
q_gt = plot(gt_velocity)
plt.clf()
plt.close()

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
lower = min(np.sqrt(q_net.U**2 + q_net.V**2).min(), np.sqrt(q_gt.U**2 + q_gt.V**2).min())
upper = max(np.sqrt(q_net.U**2 + q_net.V**2).max(), np.sqrt(q_gt.U**2 + q_gt.V**2).max())

cmap = mpl.cm.get_cmap('viridis')
normalizer = mpl.colors.Normalize(lower, upper)
colors = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)

for j, q in enumerate([q_net, q_gt]):
    plt.sca(grid[j])
    ax = grid[j]
    xyuv_im = []
    xyuv_quiver = []
    for i in [q.X, q.Y, q.U, q.V]:
        i = i.reshape(constants.GROUND_TRUTH_NX + 1, constants.GROUND_TRUTH_NX + 1)
        xyuv_im.append(i)
        xyuv_quiver.append(i[::4, ::4])

    ax.quiver(*xyuv_quiver, alpha=0.5, zorder=2)
    from dolfin.common.plotting import mesh2triang
    C = np.sqrt(xyuv_im[2]**2 + xyuv_im[3]**2).flatten()
    levels = 40
    ax.tricontourf(mesh2triang(model.mesh), C, levels, zorder=1, cmap=cmap, norm=normalizer)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

grid[-1].cax.colorbar(colors)
grid[-1].cax.toggle_label(True)

plt.tight_layout()

print(sqrt(assemble((gt_velocity - net_velocity)**2*dx)/assemble(gt_velocity**2*dx)))

plt.savefig(f"plots/paper/advection_velocities.png", dpi=400)
