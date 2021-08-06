from fenics import *
from fenics_adjoint import *
import ufl

import sympy as sp
import numpy as np


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

        bcs = DirichletBC(self.function_space, self.analytical_solution(), "on_boundary")
        solve(F == 0, u, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-12}})

        context.solver_step(u)

    def ground_truth(self, x, y):
        return 1/(1 + x**2 + y**2)

    def generate_terms(self):
        x, y = sp.var("x"), sp.var("y")
        kappa = self.ground_truth(x, y)
        u = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
        f = -(sp.diff(kappa * sp.diff(u, x), x) + sp.diff(kappa * sp.diff(u, y), y))
        return kappa, u, f.simplify()

    def analytical_solution(self):
        x, y = SpatialCoordinate(self.mesh)
        return eval(str(self.u))


class P0Model(Model):
    def __init__(self, Nx=10):
        self.mesh = UnitSquareMesh(Nx, Nx, "crossed")
        self.function_space = FunctionSpace(self.mesh, "DG", 0)

        self.test_function = TestFunction(self.function_space)

        self.kappa, self.u, self.f = self.generate_terms()

    def forward(self, term, context):
        u = Function(self.function_space)
        v = self.test_function
        x, y = SpatialCoordinate(self.mesh)

        f = eval(str(self.f))

        U = TrialFunction(self.function_space)

        h = P0Model.cell_center_distance(self.mesh)

        kappa = term(x, y)
        a = (avg(kappa) / avg(h)) * inner(jump(U), jump(v)) * dS + (kappa / h) * inner(U, v) * ds
        L = inner(f, v) * dx
        solve(a == L, u)

        context.solver_step(u)

    @staticmethod
    def dlt_to_edge_map(mesh):
        '''DLT dofs to edge indices'''
        assert mesh.geometry().dim() == mesh.topology().dim() == 2

        V = FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
        dm = V.dofmap()

        mapping = np.zeros(V.dim(), dtype='uintp')
        for cell in cells(mesh):
            cell_dofs = dm.cell_dofs(cell.index())
            for edge, edge_dof in zip(edges(cell), cell_dofs):
                mapping[edge_dof] = edge.index()

        return mapping

    @staticmethod
    def cell_center_distance(mesh):
        '''DLT function which for each has distance of the vertices sharing it'''
        assert mesh.geometry().dim() == mesh.topology().dim() == 2

        mesh.init(1, 2)
        mesh.init(1, 0)
        e2c = mesh.topology()(1, 2)
        e2v = mesh.topology()(1, 0)
        c2v = mesh.topology()(2, 0)

        x = mesh.coordinates()

        edge_values = np.zeros(mesh.num_entities(1))
        for edge in range(len(edge_values)):

            edge_cells = e2c(edge)
            # Want distance of edge midpoint to cell center
            if len(edge_cells) == 1:
                cell = edge_cells[0]
                cmp0 = np.mean(x[c2v(cell)], axis=0)
                emp = np.mean(x[e2v(edge)], axis=0)

                h = np.linalg.norm(emp - cmp0)
            # Distance of midpoints
            else:
                cell0, cell1 = e2c(edge)
                cmp0, cmp1 = np.mean(x[c2v(cell0)], axis=0), np.mean(x[c2v(cell1)], axis=0)

                h = np.linalg.norm(cmp0 - cmp1)
            # Done
            edge_values[edge] = h

        # It remains to create a DLT function
        hf = Function(FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0))
        hf.vector().set_local(edge_values[P0Model.dlt_to_edge_map(mesh)])

        return hf
