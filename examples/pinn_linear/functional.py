from fenics import *
from fenics_adjoint import *


class Functional:
    def __init__(self, sampler):
        self.sampler = sampler
        self.J = 0.

    def solver_step(self, numerical_solution, t):
        obs = self.sampler.get_observation(t)
        if obs is not None:
            self.J += assemble((numerical_solution - obs)**2*dx)


class DataWriter:
    def __init__(self, dataset):
        self.dataset = dataset

    def solver_step(self, solution, t):
        self.dataset.write(solution, t)

