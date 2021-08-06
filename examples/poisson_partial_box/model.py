from fenics import *
from fenics_adjoint import *
import ufl

import sympy as sp


class Model:
    def __init__(self, Nx=10, order=1):
        self.mesh = UnitSquareMesh(Nx, Nx)
        self.function_space = FunctionSpace(self.mesh, "CG", order)

        self.test_function = TestFunction(self.function_space)

        self.kappa, self.u, self.f = self.generate_terms()

    def forward(self, term, context):
        u = Function(self.function_space)
        v = self.test_function
        x, y = SpatialCoordinate(self.mesh)

        f = eval(str(self.f))

        F = term(x, y) * inner(grad(u), grad(v))*dx - f*v*dx

        bcs = DirichletBC(self.function_space, 0., "on_boundary")
        solve(F == 0, u, bcs)

        context.solver_step(u)

    def ground_truth(self, x, y):
        if isinstance(x, sp.Symbol):
            pi_ = sp.pi
            sin_ = sp.sin
        else:
            pi_ = pi
            sin_ = sin
        return 1 + 0.5 * sin_(2*pi_*x) * sin_(2*pi_*y)

    def generate_terms(self):
        x, y = sp.var("x"), sp.var("y")
        kappa = self.ground_truth(x, y)
        u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
        f = -(sp.diff(kappa * sp.diff(u, x), x) + sp.diff(kappa * sp.diff(u, y), y))
        return kappa, u, f

    def analytical_solution(self):
        x, y = SpatialCoordinate(self.mesh)
        return eval(str(self.u))



