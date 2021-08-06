from fenics import *
from fenics_adjoint import *


class Functional:
    def __init__(self, obs):
        self.obs = obs
        self.J = None

    def solver_step(self, numerical_solution):
        self.J = assemble((numerical_solution - self.obs)**2*dx)
