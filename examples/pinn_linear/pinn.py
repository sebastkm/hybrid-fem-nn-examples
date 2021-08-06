import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import numpy as np
import argparse

from datetime import datetime
import sympy as sp

from shared_code.torch_utils import grad, div, split_cartesian_prod

from torch_model import TorchModel
from model import DataModel, Model as FEMModel
from shared_code.data import Dataset, DeterministicSubsampler
from shared_code.experiment import Experiment
from fenics import Function

import time

torch.set_default_dtype(torch.float64)

# NOTE THAT export OMP_NUM_THREADS=1 CHANGES THE RANDOM STATE
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, dest="seed", default=int(time.time()),
                    help="The random seed to use for the model initialization. "
                         "This is used to ensure the different networks have the same initial weights.")
parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations of the optimization loop.")
parser.add_argument("--weights", type=float, nargs=4, default=[1., 1., 1., 1.],
                    help="The loss weighting. 4 weights in order: PDE term, BC term, IC term, data term.")
parser.add_argument("--name", "-n", type=str, dest="name", default=None,
                    help="Name of experiment. Default is a random UUID.")
args = parser.parse_args()
torch.manual_seed(args.seed)


pi = np.pi
sin = torch.sin
exp = torch.exp
cos = torch.cos

DATA_TYPE = torch.float64

experiment = Experiment()
experiment.seed = args.seed
experiment.args = args
experiment.torch = True

T = 0.1

data_model = DataModel(Nx=4096)
data_func = Function(data_model.function_space)
dataset = Dataset(f"data/data.xdmf", 0., T, 0.0001, precision=4, obs_func=data_func)

# Subsampler space
sampler_dt = 0.02
fem_model = FEMModel(Nx=240)
obs_func = Function(fem_model.function_space)

sampler = DeterministicSubsampler(dataset, 0., T, sampler_dt, obs_func, keep=1.0)

obs, tx = sampler.to_torch()
tx = tx[:, 1:-1]
obs = obs[:, 1:-1]
tx = torch.tensor(tx, dtype=DATA_TYPE)

tx_ic = tx[:1]
obs_ic = obs[:1]

tx = tx[1:].reshape(-1, 2)
obs = obs[1:].reshape(-1, 1)

tx_ic = tx_ic.reshape(-1, 2)
obs_ic = obs_ic.reshape(-1, 1)

# PINN collocations
delta_x = 1/240
delta_t = 0.0025

x_interior = torch.linspace(0. + delta_x, 1.0 - delta_x, 239, dtype=DATA_TYPE)
x_boundary = torch.tensor([0., 1.], dtype=DATA_TYPE)

t_interior = torch.linspace(0. + delta_t, T, 40, dtype=DATA_TYPE)
t_boundary = torch.linspace(0. + delta_t, T, 40, dtype=DATA_TYPE)
t_ic = torch.tensor([0.], dtype=DATA_TYPE)

# Format (t, x) such that they can be combined (same number of samples)
t_interior, x_interior = split_cartesian_prod(t_interior, x_interior)
t_boundary, x_boundary = split_cartesian_prod(t_boundary, x_boundary)

x_data, t_data = tx[:, 1], tx[:, 0]
x_ic, t_ic = tx_ic[:, 1], tx_ic[:, 0]

x_ic = x_ic.unsqueeze(-1)
t_ic = t_ic.unsqueeze(-1)
x_data = x_data.unsqueeze(-1)
t_data = t_data.unsqueeze(-1)

# For differentiation du/dx, du/dt.
x_interior.requires_grad = True
t_interior.requires_grad = True
x_boundary.requires_grad = True
t_boundary.requires_grad = True


model = TorchModel()
model.double()

# Format: PDE, BC, IC, Data, L2 reg
experiment.torch_optim = ([], [], [], [], [])
opt_residuals = [[], [], [], []]
def opt_cb(loss):
    opt_loss_history.append(float(loss))
    experiment.optimization_iterations += 1
    experiment.torch_optim[0].append(torch_optim[0])
    experiment.torch_optim[1].append(torch_optim[1])
    experiment.torch_optim[2].append(torch_optim[2])
    experiment.torch_optim[3].append(torch_optim[3])
    experiment.torch_optim[4].append(torch_optim[4])


optimizer = optim.LBFGS(model.parameters(), max_iter=args.maxiter, callback=opt_cb,
                        line_search_fn="strong_wolfe", history_size=1000, tolerance_change=0)

w_pde = args.weights[0]
w_bc = args.weights[1]
w_ic = args.weights[2]
w_data = args.weights[3]


torch_optim = []
opt_loss_history = []
def closure():
    global torch_optim
    optimizer.zero_grad()
    L_pde = model(x_interior, t_interior)
    L_bc = model.bc_loss(x_boundary, t_boundary)
    L_reg = 1e-8 * (model.NN(x_interior) ** 2).mean()
    L_ic = model.obs_loss(x_ic, t_ic, obs_ic).mean()
    L_data = model.obs_loss(x_data, t_data, obs).mean()

    loss = w_pde * L_pde + w_bc * L_bc + w_data * L_data + w_ic * L_ic + L_reg
    loss.backward()
    print(float(loss))
    torch_optim = [float(L_pde), float(L_bc), float(L_ic), float(L_data), float(L_reg)]
    return loss


experiment.start_time = time.time()
try:
    epochs = 1
    for epoch in range(epochs):
        print(f"Epoch = {epoch}")
        optimizer.step(closure)
except KeyboardInterrupt:
    pass
experiment.end_time = time.time()

experiment.opt_loss_history = opt_loss_history
experiment.NN = [model.u, model.NN]
if args.name is not None:
    experiment.id = args.name
experiment.save()
