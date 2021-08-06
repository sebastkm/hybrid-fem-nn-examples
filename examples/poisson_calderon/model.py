from fenics import *
from fenics_adjoint import *
import ufl

import sympy as sp


class Model:
    def __init__(self, Nx=10, order=1):
        self.mesh = UnitSquareMesh(Nx, Nx)
        self.function_space = FunctionSpace(self.mesh, "CG", order)

        self.test_function = TestFunction(self.function_space)

    def forward(self, u_data, term, context):
        u = Function(self.function_space)
        v = self.test_function
        x, y = SpatialCoordinate(self.mesh)

        F = term(x, y) * inner(grad(u), grad(v))*dx

        bcs = DirichletBC(self.function_space, u_data, "on_boundary")
        solve(F == 0, u, bcs)

        context.solver_step(u, term(x, y))

    def ground_truth(self, x, y):
        return 1 + 4 * exp(-((x - 0.5)**2 + (y - 0.5)**2)/(0.2**2))

    def boundary_condition(self):
        x, y = SpatialCoordinate(self.mesh)
        with stop_annotating():
            return project(
                exp(-((x - 1) ** 2 + (y - 0.5) ** 2) / (0.5 ** 2)) - exp(-((x - 0) ** 2 + (y - 0.5) ** 2) / (0.5 ** 2)),
                self.function_space)
