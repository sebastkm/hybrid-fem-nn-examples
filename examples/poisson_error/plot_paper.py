from fenics import *
from fenics_adjoint import *

import argparse
import ufl
import time
from numpy.random import seed

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment

from model import Model, P0Model
from functional import Functional

import re
import matplotlib.pyplot as plt


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-nx", type=int, dest="nx", default=200,
                        help="The number of FE cells to discretise the mesh into in each spatial dimension.")
parser.add_argument("--names", nargs="+", type=str, dest="names", default=["nx5", "nx10", "nx20", "nx40", "nx80"],
                    help="Name of experiments. Default is nx5 to nx80.")
args = parser.parse_args()

model = Model(Nx=args.nx, order=2)
functional = Functional(obs=model.analytical_solution())

x, y = SpatialCoordinate(model.mesh)

state_errors = []
sub_errors = []
hs = []
for name in args.names:
    experiment = Experiment(name)
    net = experiment.NN
    net.output_activation = None

    if hasattr(experiment, "nx"):
        nx = experiment.nx
    else:
        result = re.match(".*nx(\d+).*", name)
        if result:
            nx = int(result.group(1))
        else:
            raise RuntimeError(f"Unknown discretisation used for experiment {name}.")
    h = 1/nx

    model.forward(net, functional)
    state_error = sqrt(functional.J)
    sub_error = sqrt(assemble((net(x, y) - model.ground_truth(x, y)) ** 2 * dx))
    print(f"{name} | State error = ", state_error)
    print(f"{name} | Sub error = ", sub_error)

    hs.append(h)
    state_errors.append(state_error)
    sub_errors.append(sub_error)

from math import log
print("State:", [log(state_errors[i-1]/state_errors[i])/log(hs[i-1]/hs[i]) for i in range(1,len(state_errors))])
print("Sub:", [log(sub_errors[i-1]/sub_errors[i])/log(hs[i-1]/hs[i]) for i in range(1,len(sub_errors))])

plots = [
    # List of step sizes, list of errors, plot title, path of file
    [hs, state_errors, "State", "plots/poisson_state_convergence.png"],
    [hs, sub_errors, "Diffusion coefficient", "plots/poisson_sub_convergence.png"]
]

for config in plots:
    fig = plt.figure(constrained_layout=True)
    hs, errors, title, path = config

    plt.loglog(hs, errors, "x")

    rate = 2
    e0 = errors[0]
    h0 = hs[0]
    theoretical_line = [e0/(h0/hs[i])**rate for i in range(len(hs))]
    plt.plot(hs, theoretical_line, "--")

    plt.title(title)
    plt.legend(["NN P1", "O(h^2)"])
    plt.xlabel("h")
    plt.ylabel("Error")
    ax = plt.gca()
    ax.set_xscale("log", basex=2)
    ax.set_yscale("log", basey=2)
    plt.savefig(path, dpi=400)
    plt.clf()

