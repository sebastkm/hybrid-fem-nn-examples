from fenics import *
import torch

import constants


class Functional:
    def __init__(self, obs_enabled=False):
        self.sampler = None
        self.J = 0.
        self.obs_enabled = obs_enabled
        self.J_tmp = 0.
        self.num_elems = 0

    def get_obs(self, t):
        if not self.obs_enabled:
            return None
        obs = self.sampler.get_observation(t)
        return obs

    def solver_step(self, v, s, t):
        obs = self.sampler.get_observation(t)
        if obs is not None:
            se = ((v - obs)**2)
            self.num_elems += se.numel()
            self.J_tmp += se.sum()

    def compute_functional(self):
        self.J += self.J_tmp / self.num_elems
        self.num_elems = 0
        self.J_tmp = 0.
