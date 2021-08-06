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
parser.add_argument("--nx", type=int, dest="nx", default=10,
                        help="The number of FE cells to discretise the mesh into in each spatial dimension.")
parser.add_argument("--seed", type=int, dest="seed", default=int(time.time()),
                    help="The random seed to use for the model initialization. "
                         "This is used to ensure the different networks have the same initial weights.")
parser.add_argument("--snr", type=float, dest="snr", default=0,
                    help="The signal-to-noise ratio. <= 0 is no noise. Default: 0.")
parser.add_argument("--layers", type=int, nargs="+", default=[2, 10, 1],
                    help="A list of the width of the layers. " 
                         "Make sure input and output is compatible with the chosen model.")
parser.add_argument("--bias", type=bool, nargs="+", default=[True, True],
                    help="A list enabling/disabling bias for each layer (minus input layer).")
parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations of the optimization loop.")
parser.add_argument("--reg", type=float, default=1e-6, help="Regularization parameter for the L^2 regularization.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
args = parser.parse_args()

seed(args.seed)

model = Model(Nx=args.nx)

# Apply additive noise
if args.snr > 0:
    sigma = sqrt(assemble(model.analytical_solution()**2*dx)) / args.snr
    noise = Function(model.function_space)
    noise.vector()[:] = sigma * randn(noise.function_space().dim())
else:
    noise = Constant(0.)

functional = Functional(obs=model.analytical_solution(), noise=noise)

# Construct neural network
bias = args.bias
layers = args.layers
activation = sigmoid
net = ANN(layers, bias=bias, sigma=activation, init_method="uniform")

experiment = Experiment()
experiment.NN = net
experiment.seed = args.seed

model.forward(net, functional)

x, y = SpatialCoordinate(model.mesh)

J = functional.J + args.reg * assemble(grad(net(x, y))**2*dx(domain=model.mesh))
Jhat = RobustReducedFunctional(J, net.weights_ctrls())

net_test = ANN(net.layers, bias=net.bias, sigma=net.sigma)
gt_errors = []
opt_loss_history = []
def compute_real_error(*args):
    opt_loss_history.append(Jhat.functional.block_variable.checkpoint)
    with stop_annotating():
        net.opt_callback()
        net_test.set_weights(net.backup_weights_flat)
        err = sqrt(assemble((net_test(x, y) - model.ground_truth(x, y))**2*dx))
        gt_errors.append(float(err))

set_log_level(LogLevel.ERROR)

weights, loss = train(Jhat, maxiter=args.maxiter, callback=compute_real_error, ftol=1e-14, tol=1e-14, gtol=1e-14)

with stop_annotating():
    net.set_weights(weights)
Jhat(weights)

experiment.opt_loss_history = opt_loss_history
experiment.gt_errors = gt_errors
experiment.args = args
experiment.loss = loss
if args.name is not None:
    experiment.id = args.name
experiment.save()
