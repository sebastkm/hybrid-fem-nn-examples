from fenics import *
import torch
import torch.optim as optim

import argparse
import ufl
import time
from numpy.random import seed, randn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.utils import get_precision

from model import Model
from network import GTIonTerm
from functional import Functional
from data_model import DataModel
from data import Dataset, Sampler
import constants
from stimulus import get_torch_stimulus


torch.set_default_dtype(torch.float64)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--timestep", type=float, dest="dt", default=0.01,
                        help="The time step (dt) to use in the numerical model.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment.")
parser.add_argument("--dataset", type=str, dest="dataset", default="data",
                    help="Name of dataset file excluding extension. Default: data")
parser.add_argument("--stimulus", type=int, dest="stimulus", default=0,
                    help="Stimulus ID to use. Default 0.")
parser.add_argument("--t1", type=float, dest="t1", default=constants.END_TIME,
                    help="End time (T). Default is set in constants.py. Must be same as used by dataset.")
parser.add_argument("--crossed", dest="crossed", action="store_true",
                    help="Use mesh with crossed diagonals. Default is False (right diagonals).")
args = parser.parse_args()

# Create a data function (and function space)
data_model = DataModel(args.crossed)
v_func = Function(data_model.V)
s_func = Function(data_model.S)

t0, t1 = (constants.START_TIME, args.t1)
data_dt = constants.DATASET_DT
precision = constants.T_PRECISION

# Initialize our numerical model
model = Model(args.crossed)
obs_func = Function(model.V)
dt = args.dt

dataset = Dataset(f"data/{args.dataset}.xdmf", t0, t1, data_dt, precision, obs_func=(v_func, s_func))
sampler = Sampler(dataset, t0, t1, 0.01, obs_func=obs_func, s_func=Function(model.S))

stimulus = get_torch_stimulus(args.stimulus, model.V)

experiment = Experiment(args.name)
forcing, ionic_current, unet = experiment.NN
model.forcing = forcing
model.ionic_current = ionic_current
model.unet = unet


class Context:
    def __init__(self, sampler):
        self.sampler = sampler
        self.denominator = 0.
        self.numerator = 0.

        self.pred_v = Function(model.V)
        self.ref_v = Function(model.V)

    def get_obs(self, t):
        return None

    def solver_step(self, v, s, t):
        self.pred_v.vector()[:] = v.detach().numpy()

        obs = sampler.get_observation(t)
        assert obs is not None
        self.ref_v.vector()[:] = obs.detach().numpy()

        if t <= 0 or t >= 20:
            w = 0.5
        else:
            w = 1.0
        print(f"t = {t}")

        self.numerator += w * assemble((self.ref_v - self.pred_v)**2*dx)
        self.denominator += w * assemble((self.ref_v)**2*dx)


start_t = t0
v0 = sampler.get_observation(start_t)
ic_obs = sampler.get_ic_obs(start_t)
context = Context(sampler)

with torch.no_grad():
    model.forward(v0, start_t, t1, dt, stimulus, ic_obs, context)

print(f"Relative error = {sqrt(context.numerator/context.denominator)}")
print(f"Absolute error = {sqrt(context.numerator)}")

