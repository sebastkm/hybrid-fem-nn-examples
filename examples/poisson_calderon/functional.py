from fenics import *
from fenics_adjoint import *


class Functional:
    def __init__(self, target, normal):
        self.target = target
        self.n = normal
        self.J = None

    def solver_step(self, numerical_solution, coefficient):
        self.J = assemble((coefficient * inner(self.n, grad(numerical_solution)) - self.target) ** 2 * ds)

class DataWriter:
    def __init__(self, dataset):
        self.dataset = dataset

    def solver_step(self, solution, coefficient):
        self.dataset.write(solution, 1.0)
