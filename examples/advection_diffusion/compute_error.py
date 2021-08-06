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
parser.add_argument("--name", "-n", type=str, dest="name", default="nx80_1000_4x_1000_3x",
                    help="Name of experiment.")
parser.add_argument("--test-mode", type=int, dest="test_mode", default=0,
                    help="The test mode to use (integer). 0 is training set. Default: 0")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="The end time. Default is defined in constants.py.")
args = parser.parse_args()


# Create a data function (and function space)
data_function = Function(Model(Nx=constants.GROUND_TRUTH_NX).function_space)

# Initialize our numerical model
model = Model(Nx=constants.GROUND_TRUTH_NX, test_set=args.test_mode)
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
# For backwards compatibility
net.output_activation = None

x, y = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x, y), model.function_space)


class Context:
    def __init__(self, sampler):
        self.sampler = sampler
        self.precision = sampler.dataset.precision
        self.state_numerator = 0.
        self.state_denominator = 0.
        self.sub_numerator = 0.
        self.sub_denominator = 0.

    def solver_step(self, numerical_solution, t):
        t = round(t, self.precision)
        obs = sampler.get_observation(t)
        if obs is None:
            return
        print(f"t = {t}")

        if t <= 0 or t >= 5:
            w = 0.5
        else:
            w = 1.0

        num = assemble((obs - numerical_solution)**2*dx)
        den = assemble(obs**2*dx)
        self.state_numerator += w * num
        self.state_denominator += w * den

        num = assemble((model.ground_truth(obs, obs.dx(0), obs.dx(1), x, y) - net(obs, obs.dx(0), obs.dx(1), x, y))**2*dx)
        den = assemble(model.ground_truth(obs, obs.dx(0), obs.dx(1), x, y)**2*dx)
        self.sub_numerator += w * num
        self.sub_denominator += w * den


context = Context(sampler)
set_log_level(LogLevel.ERROR)
model.forward(ic, t0, t1, dt, net, context)

print("Relative state error: ", sqrt(context.state_numerator/context.state_denominator))
print("Relative sub error: ", sqrt(context.sub_numerator/context.sub_denominator))
