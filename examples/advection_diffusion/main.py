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
parser.add_argument("--nx", type=int, dest="nx", default=30,
                        help="The number of FE cells to discretise the mesh into in each spatial dimension.")
parser.add_argument("--timestep", type=float, dest="dt", default=0.1,
                        help="The time step (dt) to use in the numerical model.")
parser.add_argument("--seed", type=int, dest="seed", default=int(time.time()),
                    help="The random seed to use for the model initialization. "
                         "This is used to ensure the different networks have the same initial weights.")
parser.add_argument("--layers", type=int, nargs="+", default=[5, 30, 1],
                    help="A list of the width of the layers. " 
                         "Make sure input and output is compatible with the chosen model.")
parser.add_argument("--bias", type=bool, nargs="+", default=[True, True],
                    help="A list enabling/disabling bias for each layer (minus input layer).")
parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations of the optimization loop.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
parser.add_argument("--dataset", type=str, dest="dataset", default="data",
                    help="The name of the data file (without extension).")
parser.add_argument("--test-mode", type=int, dest="test_mode", default=0,
                    help="The test mode to use (integer). 0 is training set. Default: 0")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="The end time. Default is defined in constants.py.")
parser.add_argument("--eid", type=str, dest="eid", default=None,
                    help="Name of previous experiment to continue from.")
args = parser.parse_args()

# Set seed for reproducibility
seed(args.seed)

# Create a data function (and function space)
data_function = Function(Model(Nx=constants.GROUND_TRUTH_NX).function_space)

# Initialize our numerical model
model = Model(Nx=args.nx, test_set=args.test_mode)
obs_func = Function(model.function_space)

t0, t1 = (constants.START_TIME, args.t1)
dt = args.dt
data_dt = 0.2
precision = max(get_precision(dt), get_precision(t0), get_precision(constants.GT_TIME_STEP))

# Load data and subsample
dataset = Dataset(f"data/{args.dataset}.xdmf", t0, t1, constants.DATASET_DT, precision, obs_func=data_function)
sampler = DeterministicSubsampler(dataset, t0, t1, data_dt, obs_func, keep=1.0)
functional = Functional(sampler)

# Construct neural network
bias = args.bias
layers = args.layers
activation = ufl.tanh
net = ANN(layers, bias=bias, sigma=activation, init_method="normal")

if args.eid is not None:
    experiment = Experiment(args.eid)
    net = experiment.NN
    net.output_activation = None
else:
    experiment = Experiment()
    experiment.NN = net
    experiment.seed = seed
experiment.args = args

x, y = SpatialCoordinate(model.mesh)
ic = project(model.initial_condition(x, y), model.function_space)

model.forward(ic, t0, t1, dt, net, functional)

J = functional.J + 1e-6 * sum([assemble(w**2*dx(domain=model.mesh)) for w in net.weights_flat()])
Jhat = RobustReducedFunctional(J, net.weights_ctrls())

set_log_level(LogLevel.ERROR)

weights, loss = train(Jhat, maxiter=args.maxiter, ftol=1e-20, tol=1e-20, gtol=1e-20)

with stop_annotating():
    net.set_weights(weights)
Jhat(weights)

experiment.loss = loss
if args.name is not None:
    experiment.id = args.name
experiment.save()
