from fenics import *
import torch
import torch.optim as optim

import argparse
import ufl
import time
from numpy.random import seed, randn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
parser.add_argument("--name", "-n", type=str, dest="name", required=True,
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
sampler = Sampler(dataset, t0, t1, data_dt, obs_func=obs_func, s_func=Function(model.S))

stimulus = get_torch_stimulus(args.stimulus, model.V)

experiment = Experiment(args.name)
forcing, ionic_current, unet = experiment.NN
model.forcing = forcing
model.ionic_current = ionic_current
model.unet = unet
dt = sampler.sampler.dt

dof_coords = model.V.tabulate_dof_coordinates()
plot_idx = None
for i, coord in enumerate(dof_coords):
    if abs(coord[0] - 5) <= DOLFIN_EPS and abs(coord[1] - 20) <= DOLFIN_EPS:
        plot_idx = i
        break

if plot_idx is None:
    raise RuntimeError("Could not find index to plot.")


class Context:
    def __init__(self, sampler, plot_idx):
        self.sampler = sampler

        self.t_values = []
        self.v_values = []
        self.s_values = []

        self.obs_values = []
        self.obs_t = []

        self.obs_s_values = []

        self.ref_i_ion = []
        self.pred_i_ion = []

        self.gt_ion = GTIonTerm()

        self.plot_idx = plot_idx

    def get_obs(self, t):
        return None

    def solver_step(self, v, s, t):
        plot_idx = self.plot_idx
        self.t_values.append(float(t))
        self.v_values.append(v.detach().numpy()[plot_idx])
        self.s_values.append(s.detach().numpy()[plot_idx])

        obs = sampler.get_observation(t)
        if obs is not None:
            print(f"t = {t}")
            self.obs_t.append(float(t))
            self.obs_values.append(obs.detach().numpy()[plot_idx])

            s_obs = self.sampler.sampler.dataset.read(self.sampler.s_func, float(t), name="s")
            s0, s1, s2 = s_obs.split(deepcopy=True)
            self.obs_s_values.append((s0.vector().get_local()[plot_idx],
                                      s1.vector().get_local()[plot_idx],
                                      s2.vector().get_local()[plot_idx]))

            if t > 0:
                s_ = torch.tensor(self.obs_s_values[-1]).reshape(1, 3)
                v_ = torch.tensor(self.obs_values[-2]).reshape(1, 1)
                self.ref_i_ion.append(self.gt_ion(v_, s_))

                inp_1 = torch.tensor((self.v_values[-2] - model.v_mean) / model.v_std).reshape(1, 1)
                inp_2 = s[plot_idx].reshape(1, 3)

                pred_ion = model.ionic_current(inp_1, inp_2).detach().numpy()[0]
                pred_ion = pred_ion * model.v_std + model.v_mean
                self.pred_i_ion.append(pred_ion)


start_t = t0
v0 = sampler.get_observation(start_t)
ic_obs = sampler.get_ic_obs(start_t)
context = Context(sampler, plot_idx)

with torch.no_grad():
    model.forward(v0, start_t, t1, dt, stimulus, ic_obs, context)

plt.plot(context.t_values, context.v_values, "--", zorder=2)
plt.plot(context.obs_t, context.obs_values, zorder=1)
plt.legend(["Prediction", "Reference"], loc=1)
plt.title("Transmembrane potential at $x = (5, 20)$")
plt.xlabel("$t$")
plt.ylabel("$v$")
ax = plt.gca()
if args.stimulus == 0:
    axins = inset_axes(ax, "35%", "35%", loc=2, bbox_to_anchor=(0., 0., 1.0, 1.0),
                       bbox_transform=ax.transAxes)
    axins.plot(context.t_values, context.v_values, "--", zorder=2)
    axins.plot(context.obs_t, context.obs_values, zorder=1)
    axins.set_xlim(12.2, 12.8)
    axins.set_ylim(16, 31)
    axins.set_yticks([])
    axins.set_xticks([])
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig(f"plots/paper/cardiac_v_trained_stim{args.stimulus}.png", dpi=200)
plt.clf()
plt.plot(context.obs_t[1:], context.pred_i_ion, "--", zorder=2)
plt.plot(context.obs_t[1:], context.ref_i_ion, zorder=1)
plt.legend(["Prediction", "Reference"])
plt.title("Ionic current term at $x = (5, 20)$")
plt.xlabel("$t$")
plt.ylabel("$I_{\mathrm{ion}}$")
ax = plt.gca()
if args.stimulus == 0:
    axins = inset_axes(ax, "35%", "35%", loc=3, bbox_to_anchor=(0., 0., 1.0, 1.0), bbox_transform=ax.transAxes)
    axins.plot(context.obs_t[1:], context.pred_i_ion, "--", zorder=2)
    axins.plot(context.obs_t[1:], context.ref_i_ion, zorder=1)
    axins.set_xlim(12, 15.5)
    axins.set_ylim(-50, 95)
    axins.set_yticks([])
    axins.set_xticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.tight_layout()
plt.savefig(f"plots/paper/cardiac_i_ion_trained_stim{args.stimulus}.png", dpi=200)
plt.clf()
plt.plot(context.t_values, context.s_values)
plt.title("Predicted cell states at $x = (5, 20)$")
plt.xlabel("$t$")
plt.ylabel("$s$")
plt.tight_layout()
plt.savefig(f"plots/paper/cardiac_pred_s_trained_stim{args.stimulus}.png", dpi=200)
plt.clf()
plt.plot(context.obs_t, context.obs_s_values)
plt.title("Reference cell states at $x = (5, 20)$")
plt.xlabel("$t$")
plt.ylabel("$s$")
plt.tight_layout()
plt.savefig(f"plots/paper/cardiac_true_s_trained_stim{args.stimulus}.png", dpi=200)
plt.clf()

