from fenics import *
import ufl
import torch
import torch.nn as nn

import numpy as np
from numpy.linalg import inv

from shared_code.utils import get_precision

from hodgkin_huxley_1952 import Hodgkin_Huxley_1952
from network import ForcingTerm, IonicCurrent, GTODETerm, GTIonTerm

import constants

import time


class Model(nn.Module):
    def __init__(self, crossed=False):
        super().__init__()
        cellmodel = Hodgkin_Huxley_1952()
        self.num_states = cellmodel.num_states()

        self.forcing = ForcingTerm()
        self.ionic_current = IonicCurrent()
        self.unet = None

        self.v_mean = -75
        self.v_std = 75 + 40

        diagonal = "crossed" if crossed else "right"
        self.mesh = RectangleMesh(Point((0, 0)), Point(constants.OMEGA_DIMENSIONS), *constants.FEM_CELLS, diagonal)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.S = VectorFunctionSpace(self.mesh, "CG", 1, dim=self.num_states)
        self._cache = {}

        self.idx_to_image = None
        self.inv_idx_to_image = None

        self.i_ion_terms = []

        self.I_s = None

    def forward(self, v0, t0, t1, dt, stimulus, ic_obs, context, s=None):
        if ic_obs is None:
            s0 = torch.zeros(*v0.shape[:1], self.num_states, device=v0.device)
        else:
            assert self.unet is not None, "UNet is not enabled."
            s0 = self.image_to_fem(self.unet(self.fem_to_image(ic_obs)))
            s0 = s0.squeeze(0)

        if s is not None:
            s0 = s

        t = torch.tensor(t0)
        dt_ = torch.tensor(dt)
        dt = torch.tensor(dt, device=v0.device)

        context.solver_step(v0, s0, t)

        while round(float(t), constants.T_PRECISION) < t1:
            t = t + dt_
            v, s = self.solve(v0, s0, t, dt, dt_, stimulus)
            context.solver_step(v, s, t)

            obs = context.get_obs(t)
            if obs is not None:
                v0 = obs
            else:
                v0 = v
            s0 = s

    def solve(self, v0, s0, t, dt, dt_, stimulus):
        v_flat = v0.view(-1, 1)
        s_flat = s0.view(-1, self.num_states)

        v_flat_norm = (v_flat - self.v_mean) / self.v_std

        f = self.forcing(v_flat_norm, s_flat)

        s_flat = s_flat + dt * f

        I_ion = self.ionic_current(v_flat_norm, s_flat)
        I_ion = I_ion * self.v_std + self.v_mean
        I_ion = I_ion.view(v0.shape)

        # Stimulus
        if self.I_s is None:
            self.I_s = torch.zeros(v0.shape, device=v0.device)
        I_s = self.I_s.zero_()
        I_s = stimulus(I_s, t)

        inv_pde, mass = self.get_operators(dt_, v0.device)

        rhs = torch.matmul(mass, v0) + dt * torch.matmul(mass, I_s) - dt * torch.matmul(mass, I_ion)
        v = torch.matmul(inv_pde, rhs)

        s = s_flat.view(s0.shape)
        v = v.view(v0.shape)

        return v, s

    def get_operators(self, dt, device):
        key = round(float(dt), constants.T_PRECISION)
        if key not in self._cache:
            V = self.V
            # Define functions
            q = TestFunction(V)
            v_ = TrialFunction(V)

            # Define model parameters
            dt = Constant(dt.cpu())  # ms
            chi = 140  # mm^-1
            C_m = 0.01  # uF/mm^2
            M = as_tensor([[0.174 / (chi * C_m), 0], [0, 0.174 / (chi * C_m)]])

            # Define variational forms
            operator = v_ * q * dx + dt * inner(M * grad(v_), grad(q)) * dx
            operator_matrix = assemble(operator).array()
            operator_matrix = inv(operator_matrix)
            operator_matrix = torch.tensor(operator_matrix, device=device)

            mass = v_ * q * dx
            mass_matrix = assemble(mass).array()
            mass_matrix = torch.tensor(mass_matrix, device=device)

            self._cache[key] = (operator_matrix, mass_matrix)
        return self._cache[key]

    def fem_to_image(self, ic_obs):
        if self.idx_to_image is None:
            coords = self.V.tabulate_dof_coordinates()
            self.idx_to_image = np.lexsort((coords[:, 1], coords[:, 0]))
            self.inv_idx_to_image = self.idx_to_image.argsort()
        return ic_obs.squeeze(-1)[:, self.idx_to_image].reshape(-1, 16, 16).unsqueeze(0)

    def image_to_fem(self, image):
        return image.reshape(image.shape[0], image.shape[1], 16 * 16)[:, :, self.inv_idx_to_image].transpose(1, 2)
