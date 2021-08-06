from fenics import *
from fenics_adjoint import *


class Functional:
    def __init__(self, sampler):
        self.sampler = sampler
        self.J = 0.
        self.u_hats = {}
        self.precision = sampler.dataset.precision

    def solver_step(self, numerical_solution, t):
        obs = self.sampler.get_observation(t)
        self.u_hats[round(t, self.precision)] = Control(numerical_solution)
        if obs is not None:
            self.J += assemble((numerical_solution - obs)**2*dx)


class DataWriter:
    def __init__(self, dataset):
        self.dataset = dataset

    def solver_step(self, solution, t):
        t = round(t, self.dataset.precision)
        if t in self.dataset.time_points:
            self.dataset.write(solution, t)
            print(f"Wrote data at t = {t}")

