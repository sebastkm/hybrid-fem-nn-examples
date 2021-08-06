from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed

from ufl_dnn.neural_network import ANN, sigmoid, ELU

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment

from model import Model, P0Model
from functional import Functional


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-nx", type=int, dest="nx", default=10,
                        help="The number of FE cells to discretise the mesh into in each spatial dimension.")
parser.add_argument("--seed", type=int, dest="seed", default=int(time.time()),
                    help="The random seed to use for the model initialization. "
                         "This is used to ensure the different networks have the same initial weights.")
parser.add_argument("--layers", type=int, nargs="+", default=[2, 10, 1],
                    help="A list of the width of the layers. " 
                         "Make sure input and output is compatible with the chosen model.")
parser.add_argument("--bias", type=bool, nargs="+", default=[True, True],
                    help="A list enabling/disabling bias for each layer (minus input layer).")
parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations of the optimization loop.")
parser.add_argument("--model", type=str.lower, default="p1", choices=["p0", "p1", "p2"], help="Maximum iterations of the optimization loop.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
args = parser.parse_args()

seed(args.seed)
if args.model == "p0":
    model = P0Model(Nx=args.nx)
elif args.model == "p2":
    model = Model(Nx=args.nx, order=2)
else:
    model = Model(Nx=args.nx)
functional = Functional(obs=model.analytical_solution())

# Construct neural network
bias = args.bias
layers = args.layers
activation = sigmoid
net = ANN(layers, bias=bias, sigma=activation, init_method="uniform")
net.output_activation = None

experiment = Experiment()
experiment.NN = net
experiment.seed = args.seed
experiment.args = args

model.forward(net, functional)

J = functional.J
Jhat = ReducedFunctional(J, net.weights_ctrls())

x, y = SpatialCoordinate(model.mesh)

set_log_level(LogLevel.ERROR)

opt_loss_history = []
def cb(*args):
    opt_loss_history.append(Jhat.functional.block_variable.checkpoint)

weights, loss = train(Jhat, callback=cb, maxiter=args.maxiter, ftol=0, tol=0, gtol=0)

with stop_annotating():
    net.set_weights(weights)

print(sqrt(assemble( (net(x, y) - model.ground_truth(x, y))**2 * dx)))

experiment.opt_loss_history = opt_loss_history
experiment.loss = loss
if args.name is not None:
    experiment.id = args.name
experiment.save()
