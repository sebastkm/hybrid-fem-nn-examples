from fenics import *
import torch
import torch.optim as optim

import argparse
import ufl
import time
import copy
from numpy.random import seed, randn
import numpy as np

from ufl_dnn.neural_network import ANN, sigmoid

from shared_code.training import train, RobustReducedFunctional
from shared_code.experiment import Experiment
from shared_code.utils import get_precision
from functools import partial

from model import Model
from functional import Functional
from data_model import DataModel
from data import Dataset, Sampler
import constants
from stimulus import get_torch_stimulus


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--timestep", nargs="+", type=float, dest="dt", default=[0.01],
                        help="The time step (dt) to use in the numerical model. "
                             "Can specify multiple ones to train on multiple timesteps simultaneously. "
                             "Default: 0.01.")
parser.add_argument("--seed", type=int, dest="seed", default=int(time.time()),
                    help="The random seed to use for the model initialization. "
                         "This is used to ensure the different networks have the same initial weights.")
parser.add_argument("--maxiter-adam", dest="maxiter_adam", type=int, default=100, help="Maximum iterations of the Adam optimization.")
parser.add_argument("--maxiter-bfgs", dest="maxiter_bfgs", type=int, default=100, help="Maximum iterations of the BFGS optimization.")
parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate used by the Adam optimizer during training.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
parser.add_argument("--eid", type=str, dest="eid", default=None,
                    help="Name of experiment to reuse.")
parser.add_argument("--dataset", nargs="+", type=str, dest="dataset", default=["data"],
                    help="Name of dataset file excluding extension. Can specify multiple datasets. Default: data.")
parser.add_argument("--stimulus", nargs="+", type=int, dest="stimulus", default=[0],
                    help="Stimulus ID to use. Can be multiple stimuli. Default 0.")
parser.add_argument("--t1", nargs="+", type=float, dest="t1", default=[constants.END_TIME],
                    help="End time (T). Default is set in constants.py. Must be same as used by dataset(s).")
parser.add_argument("--feed-obs", dest="feed_obs", action="store_true",
                    help="Feed observations as inputs whenever available.")
parser.add_argument("--crossed", dest="crossed", action="store_true",
                    help="Use mesh with crossed diagonals. Default is False (right diagonals).")
args = parser.parse_args()

# Set seed for reproducibility
seed(args.seed)
torch.manual_seed(args.seed)

# Create a data function (and function space)
data_model = DataModel(args.crossed)
v_func = Function(data_model.V)
s_func = Function(data_model.S)

t0 = constants.START_TIME
data_dt = constants.DATASET_DT
precision = constants.T_PRECISION

# Initialize our numerical model
model = Model(args.crossed)
obs_func = Function(model.V)

# Setup different stimulus datasets
datasets = []
for i, dataset_file in enumerate(args.dataset):
    t1 = args.t1[0] if len(args.t1) == 1 else args.t1[i]
    dataset = Dataset(f"data/{dataset_file}.xdmf", t0, t1, data_dt, precision, obs_func=(v_func, s_func))
    sampler = Sampler(dataset, t0, t1, 0.1, obs_func=obs_func)

    stim_id = args.stimulus[0] if len(args.stimulus) == 1 else args.stimulus[i]
    stimulus = get_torch_stimulus(stim_id, model.V)
    datasets.append((sampler, stimulus, t1))

training_sets = []
if len(args.dt) == 1:
    dt = args.dt[0]
    training_sets = [(*dataset, dt) for dataset in datasets]
elif len(args.dt) == len(datasets):
    training_sets = [(*dataset, dt) for dataset, dt in zip(datasets, args.dt)]
elif len(datasets) == 1:
    dataset = datasets[0]
    training_sets = [(*dataset, dt) for dt in args.dt]
else:
    raise ValueError("The number of timestep sizes must match the number of datasets if using multiple datasets.")


if args.eid is not None:
    device = torch.device("cpu")

    experiment = Experiment(args.eid)
    model.forcing = experiment.NN[0]
    model.ionic_current = experiment.NN[1]
    model.unet = experiment.NN[2]

    model = model.to(device=device)
else:
    experiment = Experiment()
    experiment.NN = [model.forcing, model.ionic_current, model.unet]
    experiment.seed = args.seed
experiment.args = args

batch_size = 16

best_loss = 1e+20
best_model = Model(args.crossed)
def create_closure(configs, optimizer):
    def closure():
        global best_loss, best_model
        optimizer.zero_grad()
        functional = Functional(obs_enabled=args.feed_obs)
        for config in configs:
            t0 = config["t0"]
            t1 = config["t1"]
            sampler, stimulus, _, dt = training_sets[config["training_set"]]

            v0 = sampler.get_observation(t0)
            ic_obs = sampler.get_ic_obs(t0)
            functional.sampler = sampler

            model.forward(v0, t0, t1, dt, stimulus, ic_obs, functional)
            functional.compute_functional()
        loss = functional.J
        print(f"Loss = {loss} | Best = {best_loss}")
        if torch.isnan(loss).any():
            return torch.tensor(1e+20)

        if loss < best_loss:
            best_loss = float(loss)
            best_model.forcing = copy.deepcopy(model.forcing)
            best_model.ionic_current = copy.deepcopy(model.ionic_current)
            best_model.unet = copy.deepcopy(model.unet)
        loss.backward()
        return loss
    return closure


def random_configs(batch_size, training_sets):
    configs = []
    for batch in range(batch_size):
        training_set = np.random.choice(len(training_sets))
        sampler, _, t1, _ = training_sets[training_set]
        dt = sampler.sampler.dt

        num_choices = int(round((t1 - t0)/dt, 0) - 4)
        start_i = np.random.choice(num_choices)

        config = {
            "t0": t0 + 3 * dt + start_i * dt,
            "t1": t1,
            "training_set": training_set
        }
        configs.append(config)
    return configs


def semi_full_configs(training_sets):
    configs = []
    for i, training_set in enumerate(training_sets):
        sampler, _, t1, _ = training_set
        dt = sampler.sampler.dt
        config = {
            "t0": t0 + 3 * dt,
            "t1": t1,
            "training_set": i
        }
        configs.append(config)
    return configs


def full_configs(training_sets):
    configs = []
    for i, training_set in enumerate(training_sets):
        _, _, t1, _ = training_set
        config = {
            "t0": t0,
            "t1": t1,
            "training_set": i
        }
        configs.append(config)
    return configs


optimizer = optim.Adam(model.parameters(), lr=args.lr)
configs = full_configs(training_sets)

try:
    for i in range(args.maxiter_adam):
        print(f"Epoch = {i}")
        optimizer.step(create_closure(configs, optimizer))
except KeyboardInterrupt:
    print("Interrupted!")

if args.maxiter_bfgs > 0:
    try:
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.maxiter_bfgs,
                                max_eval=1_000_000_000, history_size=100, tolerance_grad=1e-12,
                                tolerance_change=1e-12, line_search_fn="strong_wolfe")
        #from shared_code.scipy_optim import ScipyOptimizer
        #optimizer = ScipyOptimizer(model.parameters(), method="L-BFGS-B", maxiter=args.maxiter_bfgs, disp=True)
        optimizer.step(create_closure(configs, optimizer))
    except KeyboardInterrupt:
        print("Interrupted!")

experiment.NN = [best_model.forcing, best_model.ionic_current, best_model.unet]

if args.name is not None:
    experiment.id = args.name
experiment.save()

print("Saved experiment to file.")
