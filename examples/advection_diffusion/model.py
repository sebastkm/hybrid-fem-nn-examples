from fenics import *
from fenics_adjoint import *
import ufl

import sympy as sp

from shared_code.utils import get_precision


class Model:
    def __init__(self, Nx=10, order=1, test_set=0):
        self.mesh = UnitSquareMesh(Nx, Nx)
        self.function_space = FunctionSpace(self.mesh, "CG", order)

        self.test_function = TestFunction(self.function_space)
        self.D = Constant(0.1)
        self.test_set = test_set

    def forward(self, u0, t0, t1, dt, term, context):
        u = Function(self.function_space)
        v = self.test_function
        x, y = SpatialCoordinate(self.mesh)

        dt_ = Constant(dt)
        theta = 0.5
        a = self.D * (theta * inner(grad(u), grad(v)) + (1 - theta) * inner(grad(u0), grad(v))) * dx
        a += ((1 - theta) * term(u0, u0.dx(0), u0.dx(1), x, y) + theta * term(u, u.dx(0), u.dx(1), x, y)) * v * dx(degree=2)
        F = (u - u0)/dt_ * v * dx + a

        u.assign(u0)
        t = t0
        t_ = Constant(t)
        bcs_expr = self.bcs_expr(t_)
        bcs = self.bcs(bcs_expr)

        precision = max(get_precision(t), get_precision(dt))

        while round(t, precision) < t1:
            t += dt
            t_.assign(t)
            solve(F == 0, u, bcs)
            u0.assign(u)
            context.solver_step(u, t)

    def ground_truth(self, u, u_x, u_y, x, y):
        velocity = as_vector([sin(pi * x) * cos(pi * y), -cos(pi * x) * sin(pi * y)])
        return velocity[0].dx(0) * u + velocity[0] * u_x + velocity[1].dx(1) * u + velocity[1] * u_y

    def initial_condition(self, x, y):
        return Constant(0.)

    def bcs_expr(self, t):
        if self.test_set == 1:
            return Expression("2 * sin(0.5 * t) * sin(0.5 * t)", t=t, degree=1)
        return Expression("sin(t) * sin(t)", t=t, degree=1)

    def bcs(self, expr):
        return DirichletBC(self.function_space, expr, "on_boundary && near(x[0], 0)")


class DataModel(Model):
    def forward(self, u0, t0, t1, dt, term, context):
        from gryphon import ESDIRK

        u = Function(self.function_space)
        v = self.test_function
        x, y = SpatialCoordinate(self.mesh)
        D = self.D

        bc_expr = self.bcs_expr(t0)
        bcs = self.bcs(bc_expr)
        rhs = - D * inner(grad(u), grad(v)) * dx - term(u, u.dx(0), u.dx(1), x, y)*v*dx

        u.assign(u0)
        t = t0

        precision = max(get_precision(t), get_precision(dt))

        while round(t, precision) < t1:
            obj = ESDIRK([t, t + dt], u, rhs, bcs=[bcs], tdfBC=[bc_expr])
            obj.parameters["verbose"] = True
            obj.parameters['timestepping']['dt'] = dt
            obj.parameters["timestepping"]["adaptive"] = False
            obj.solve()

            t += dt
            context.solver_step(u, t)
