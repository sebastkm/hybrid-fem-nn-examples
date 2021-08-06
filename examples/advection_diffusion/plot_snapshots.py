from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed, randn

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.data import Dataset, DeterministicSubsampler
from shared_code.utils import get_precision

from model import Model, DataModel
from functional import Functional
import constants


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, dest="dataset", default="data",
                        help="The name of the data file (without extension).")
parser.add_argument("--time", type=float, dest="t", default=1.0,
                        help="The time to plot the state at.")
parser.add_argument("--name", "-n", type=str, dest="name",
                    help="Name of experiment.")
parser.add_argument("--test", dest="test", action="store_true",
                    help="Use test set.")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="The end time. Default is defined in constants.py.")
parser.add_argument("--base", dest="base", action="store_true",
                    help="Include plot of only PDE solution (net = 0).")
args = parser.parse_args()


# Create a data function (and function space)
data_function = Function(Model(Nx=constants.GROUND_TRUTH_NX).function_space)

# Initialize our numerical model
model = Model(Nx=constants.GROUND_TRUTH_NX, test_set=args.test)
obs_func = Function(model.function_space)

t0, t1 = (constants.START_TIME, args.t1)
dt = constants.GT_TIME_STEP
precision = max(get_precision(dt), get_precision(t0), get_precision(constants.GT_TIME_STEP))

# Load data and subsample
dataset = Dataset(f"data/{args.dataset}.xdmf", t0, t1, constants.DATASET_DT, precision, obs_func=data_function)
sampler = DeterministicSubsampler(dataset, t0, t1, constants.DATASET_DT, obs_func, keep=1.0)

# Construct neural network
experiment = Experiment(args.name)
net = experiment.NN

x, y = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x, y), model.function_space)

class Context:
    def __init__(self, sampler, t):
        self.sampler = sampler
        self.numerical_solution = None
        self.precision = sampler.dataset.precision
        self.t = round(t, self.precision)

    def solver_step(self, numerical_solution, t):
        t = round(t, self.precision)
        if t == self.t:
            self.numerical_solution = Control(numerical_solution)


context = Context(sampler, args.t)
model.forward(ic, t0, args.t, dt, net, context)

obs = sampler.get_observation(args.t)
x, y = SpatialCoordinate(model.mesh)

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

if args.base:
    fig_width = 10.0
    ncols = 3
    sca_offset = 1

    context2 = Context(sampler, args.t)
    ic = project(model.initial_condition(x, y), model.function_space)
    model.forward(ic, t0, args.t, dt, lambda *args: Constant(0.), context2)
    base_numerical_solution = context2.numerical_solution.tape_value()
else:
    fig_width = 7.0
    ncols = 2
    sca_offset = 0

fig = plt.figure(figsize=(fig_width, 3), constrained_layout=True)

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,ncols),
                 axes_pad=0.35,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )


num_sol_func = context.numerical_solution.tape_value()
# Compute limits
lower = min(num_sol_func.vector().min(), obs.vector().min())
upper = max(num_sol_func.vector().max(), obs.vector().max())
if args.base:
    lower = min(base_numerical_solution.vector().min(), lower)
    upper = max(base_numerical_solution.vector().max(), upper)

cmap = mpl.cm.get_cmap('viridis')
normalizer = mpl.colors.Normalize(lower, upper)
colors = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)

if args.base:
    plt.sca(grid[0])
    plt.title(f"No advection, t = {args.t}")
    plot(base_numerical_solution, cmap=cmap, norm=normalizer)

plt.sca(grid[sca_offset + 0])
plt.title(f"Predicted state, t = {args.t}")
plot(num_sol_func, cmap=cmap, norm=normalizer)

plt.sca(grid[sca_offset + 1])
plt.title(f"True state, t = {args.t}")
plot(obs, cmap=cmap, norm=normalizer)

grid[-1].cax.colorbar(colors)
grid[-1].cax.toggle_label(True)

plt.tight_layout()
test = "_test" if args.test else ""
base = "_base" if args.base else ""
plt.savefig(f"plots/paper/advection_state{test}{base}_t{str.replace(str(args.t), '.', '_')}.png", dpi=400)

print("Relative state error at t: ", sqrt(assemble((context.numerical_solution.tape_value() - obs)**2*dx)/assemble(obs**2*dx)))
u = context.numerical_solution.tape_value()
print("Relative sub error at t: ", sqrt(assemble((model.ground_truth(u, u.dx(0), u.dx(1), x, y) - net(u, u.dx(0), u.dx(1), x, y))**2*dx)/assemble(u**2*dx)))

