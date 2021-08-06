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
parser.add_argument("--dataset", type=str, dest="dataset", default="time",
                        help="The name of the data file (without extension).")
parser.add_argument("--name", "-n", type=str, dest="name",
                    help="Name of experiment.")
parser.add_argument("--test", dest="test", action="store_true",
                    help="Use test set.")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="The end time. Default is defined in constants.py.")
args = parser.parse_args()

# Create a data function (and function space)
data_function = Function(Model(Nx=constants.GROUND_TRUTH_NX).function_space)

# Initialize our numerical model
model = Model(Nx=30, test_set=args.test)
obs_func = Function(model.function_space)

t0, t1 = (constants.START_TIME, args.t1)
dt = 0.1
precision = max(get_precision(dt), get_precision(t0), get_precision(constants.GT_TIME_STEP))

# Load data and subsample
dataset = Dataset(f"data/{args.dataset}.xdmf", t0, t1, constants.DATASET_DT, precision, obs_func=data_function)
sampler = DeterministicSubsampler(dataset, t0, t1, constants.DATASET_DT, obs_func, keep=1.0)

# Construct neural network
experiment = Experiment(args.name)
net = experiment.NN
net.output_activation = None

x, y = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x, y), model.function_space)

class Context:
    def __init__(self, sampler):
        self.sampler = sampler
        self.errors = []
        self.ts = []
        self.precision = sampler.dataset.precision

    def solver_step(self, numerical_solution, t):
        obs = self.sampler.get_observation(t)
        if obs is not None:
            self.ts.append(round(t, self.precision))
            self.errors.append(sqrt(assemble((numerical_solution - obs)**2*dx)/assemble(obs**2*dx)))

context = Context(sampler)
with stop_annotating():
    model.forward(ic, t0, t1, dt, net, context)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.plot(context.ts, context.errors)

plt.axvline(x=constants.END_TIME, color="r", dashes=[2, 2])

plt.ylabel("Relative error")
plt.xlabel("t")
plt.tight_layout()
plt.savefig("plots/paper/advection_time_extrapolation.png", dpi=400)
