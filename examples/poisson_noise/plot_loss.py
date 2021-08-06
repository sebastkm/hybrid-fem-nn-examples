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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--names", "-n", type=str, nargs=2, dest="names",
                    help="Names of experiments to plot.")
parser.add_argument("--legends", type=str, nargs=2, dest="legends",
                    help="Legends of experiments on plot.")
args = parser.parse_args()

plt.figure()

losses = []
for name in args.names:
    experiment = Experiment(name)
    print(experiment.args)
    plt.semilogy(experiment.gt_errors)

plt.legend(args.legends)
plt.xlabel("Iterations")
plt.ylabel("Prediction error")
plt.tight_layout()
plt.savefig("plots/plot_loss.png", dpi=400, bbox_inches='tight')
