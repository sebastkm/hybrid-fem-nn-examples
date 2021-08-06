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
parser.add_argument("--seed", type=int, dest="seed", default=int(time.time()),
                    help="The random seed to use for the model initialization. "
                         "This is used to ensure the different networks have the same initial weights.")
parser.add_argument("--layers", type=int, nargs="+", default=[1, 30, 1],
                    help="A list of the width of the layers. " 
                         "Make sure input and output is compatible with the chosen model.")
parser.add_argument("--bias", type=bool, nargs="+", default=[True, True],
                    help="A list enabling/disabling bias for each layer (minus input layer).")
parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations of the optimization loop.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
args = parser.parse_args()

# Set seed for reproducibility
seed(args.seed)

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

# Construct neural network
bias = args.bias
layers = args.layers
activation = ufl.tanh
# Initialize weights with uniform [0, 1) distribution to ensure net >= 0 initially.
net = ANN(layers, bias=bias, sigma=activation, init_method="uniform")

experiment = Experiment()
experiment.NN = net
experiment.seed = args.seed
experiment.args = args

x, *_ = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x), model.function_space)

model.forward(ic, t0, t1, dt, net, functional)

J = functional.J + 1e-8 * assemble(net(x)**2*dx(domain=model.mesh))
Jhat = ReducedFunctional(J, net.weights_ctrls())

set_log_level(LogLevel.ERROR)

experiment.start_time = time.time()
weights, loss = train(Jhat, maxiter=args.maxiter, ftol=1e-20, tol=1e-20, gtol=1e-20)
experiment.end_time = time.time()

with stop_annotating():
    net.set_weights(weights)
Jhat(weights)

experiment.loss = loss
if args.name is not None:
    experiment.id = args.name
experiment.save()
