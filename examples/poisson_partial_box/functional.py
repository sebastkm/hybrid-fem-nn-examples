from fenics import *
from fenics_adjoint import *


class Functional:
    def __init__(self, obs, measure):
        self.obs = obs
        self.J = None
        self.dx = measure

    def solver_step(self, numerical_solution):
        self.J = assemble((numerical_solution - self.obs)**2*self.dx)
