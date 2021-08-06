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


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations of the optimization loop.")
parser.add_argument("--reg", type=str, default="l2", help="Either l2 or h1 regularization. Default l2.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
args = parser.parse_args()

# Create a data function (and function space)
data_function = Function(Model(Nx=4096).function_space)

# Initialize our numerical model
model = Model(Nx=240)
obs_func = Function(model.function_space)

t0, t1 = (0., 0.1)
dt = 0.0025
precision = 4
sampler_dt = 0.02

# Load data and subsample
dataset = Dataset("data/data.xdmf", t0, t1, 0.0001, precision, obs_func=data_function)
sampler = DeterministicSubsampler(dataset, t0, t1, sampler_dt, obs_func, keep=1.0)
functional = Functional(sampler)

approximator = Function(model.function_space)
approximator.vector()[:] = 1
term = lambda *args: approximator

experiment = Experiment()
experiment.fem = True
experiment.args = args

x, *_ = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x), model.function_space)

model.forward(ic, t0, t1, dt, term, functional)

J = functional.J
if args.reg == "h1":
    J += 1e-8 * assemble(grad(approximator)**2*dx(domain=model.mesh))
else:
    J += 1e-8 * assemble(approximator ** 2 * dx(domain=model.mesh))
Jhat = RobustReducedFunctional(J, [Control(approximator)])

set_log_level(LogLevel.ERROR)

experiment.start_time = time.time()
weights, loss = train(Jhat, maxiter=args.maxiter, ftol=1e-20, tol=1e-20, gtol=1e-20)
experiment.end_time = time.time()

print("J = ", Jhat(weights))

experiment.NN = weights.vector().get_local()
experiment.loss = loss
if args.name is not None:
    experiment.id = args.name
experiment.save()
