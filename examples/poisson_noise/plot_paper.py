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

# Set seed to reproduce noise.
seed(experiment.seed)

model = Model(Nx=80)

# Apply additive noise
if experiment.args.snr > 0:
    sigma = sqrt(assemble(model.analytical_solution()**2*dx)) / experiment.args.snr
    noise = Function(model.function_space)
    noise.vector()[:] = sigma * randn(noise.function_space().dim())
else:
    noise = Constant(0.)

functional = Functional(obs=model.analytical_solution(), noise=noise)
model.forward(net, functional)
num_sol = Control(functional.num_sol)

x, y = SpatialCoordinate(model.mesh)
proj_net = project(net(x, y), model.function_space)
proj_gt = project(model.ground_truth(x, y), model.function_space)

import matplotlib.pyplot as plt
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


plt.sca(grid[0])
colors = plot(proj_net, vmin=lower, vmax=upper)

plt.sca(grid[1])
colors = plot(proj_gt, vmin=lower, vmax=upper)

grid[-1].cax.colorbar(colors)
grid[-1].cax.toggle_label(True)

plt.tight_layout()
plt.savefig("plots/poisson_noise_sub.png", dpi=400)

# State
#

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


num_sol_func = num_sol.tape_value()
proj_analytical = project(model.analytical_solution() + functional.noise, model.function_space)
# Compute limits
lower = min(num_sol_func.vector().min(), proj_analytical.vector().min())
upper = max(num_sol_func.vector().max(), proj_analytical.vector().max())


plt.sca(grid[0])
colors = plot(num_sol_func, vmin=lower, vmax=upper)

plt.sca(grid[1])
colors = plot(proj_analytical, vmin=lower, vmax=upper)

grid[-1].cax.colorbar(colors)
grid[-1].cax.toggle_label(True)

plt.tight_layout()
plt.savefig("plots/poisson_noise_state.png", dpi=400)

print("State error = ", sqrt(assemble((num_sol.tape_value() - model.analytical_solution())**2*dx)/assemble(model.analytical_solution()**2*dx)))
print("Network error = ", sqrt(assemble((net(x, y) - model.ground_truth(x, y))**2*dx)/assemble(model.ground_truth(x, y)**2*dx)))
