from fenics import *
from fenics_adjoint import *


class Functional:
    def __init__(self, obs, noise):
        self.obs = obs
        self.J = None
        self.noise = noise
        self.num_sol = None

    def solver_step(self, numerical_solution):
        self.J = assemble((numerical_solution - self.obs - self.noise)**2*dx)
        self.num_sol = numerical_solution
