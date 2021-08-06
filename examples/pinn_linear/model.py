from fenics import *
from fenics_adjoint import *
import ufl

import sympy as sp

from shared_code.utils import get_precision


class Model:
    def __init__(self, Nx=10, order=1):
        self.mesh = UnitIntervalMesh(Nx)
        self.function_space = FunctionSpace(self.mesh, "CG", order)

        self.test_function = TestFunction(self.function_space)

    def forward(self, u0, t0, t1, dt, term, context):
        u = Function(self.function_space)
        v = self.test_function
        x, *_ = SpatialCoordinate(self.mesh)

        bcs = self.bcs()
        dt_ = Constant(dt)
        theta = 0.5
        a = term(x) * (theta * inner(grad(u), grad(v)) + (1 - theta) * inner(grad(u0), grad(v))) * dx
        F = (u - u0)/dt_ * v * dx + a

        u.assign(u0)
        t = t0

        precision = max(get_precision(t), get_precision(dt))

        while round(t, precision) < t1:
            solve(F == 0, u, bcs)
            u0.assign(u)
            t += dt
            context.solver_step(u, t)

    def ground_truth(self, x):
        return 2*x + 1

    def initial_condition(self, x):
        return x * (1 - x)

    def bcs(self):
        return DirichletBC(self.function_space, 0., "on_boundary")


class DataModel(Model):
    def forward(self, u0, t0, t1, dt, term, context):
        from gryphon import ESDIRK

        u = Function(self.function_space)
        v = self.test_function
        x, *_ = SpatialCoordinate(self.mesh)

        bcs = self.bcs()
        rhs = - term(x) * inner(grad(u), grad(v)) * dx

        u.assign(u0)
        t = t0

        precision = max(get_precision(t), get_precision(dt))

        while round(t, precision) < t1:
            obj = ESDIRK([t, t + dt], u, rhs, bcs=[bcs])
            obj.parameters["verbose"] = True
            obj.parameters['timestepping']['dt'] = dt
            obj.parameters["timestepping"]["adaptive"] = False
            obj.solve()

            t += dt
            context.solver_step(u, t)
